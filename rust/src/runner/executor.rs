use std::collections::HashSet;
use std::process::Stdio;
use std::sync::Arc;

use itertools::izip;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::{mpsc, Mutex, Notify};
use tokio::task::JoinHandle;
use uuid::Uuid;

use crate::comm;

#[derive(Serialize, Debug)]
#[serde(tag = "t", content = "c")]
enum Request<'a> {
    Init(Init<'a>),
    Shutdown(Shutdown),
    AddRequest(AddRequest<'a>),
    CancelRequest(CancelRequest),
    BatchPrefill(BatchPrefill<'a>),
    BatchDecode(BatchDecode<'a>),
}

#[derive(Serialize, Debug)]
struct Init<'a> {
    use_fake: bool,
    model_path: &'a str,
    dtype_str: &'a str,
    block_len: u32,
    kvpool_capacity: u32,
}

#[derive(Serialize, Debug)]
struct Shutdown {}

#[derive(Serialize, Debug)]
struct AddRequest<'a> {
    reqid: Uuid,
    input_ids: &'a [u32],
    gencfg: &'a comm::GenerationConfig,
}

#[derive(Serialize, Debug)]
struct CancelRequest {
    reqid: Uuid,
}

#[derive(Serialize, Debug)]
struct BatchPrefill<'a> {
    reqids: &'a [Uuid],
}

#[derive(Serialize, Debug)]
struct BatchDecode<'a> {
    reqids: &'a [Uuid],
}

#[derive(Deserialize, Debug)]
pub struct TextGenerationChunkResponse {
    pub token_ids: Vec<u32>,
    pub finish_reasons: Vec<comm::FinishReason>,
}

pub struct ExecutorSubprocess {
    stdin: ChildStdin,
    stdout: ChildStdout,
}

impl ExecutorSubprocess {
    pub fn spawn(gpu_uuid: Uuid) -> anyhow::Result<(Child, Self)> {
        let mut child = Command::new("python")
            .args(["-m", "punica_runner.gpu_executor"])
            .env("CUDA_VISIBLE_DEVICES", format!("GPU-{}", gpu_uuid))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true)
            .spawn()?;
        let stdin = child.stdin.take().unwrap();
        let stdout = child.stdout.take().unwrap();
        Ok((child, Self { stdin, stdout }))
    }

    async fn write_msg(&mut self, msg: &Request<'_>) -> anyhow::Result<()> {
        let bin = rmp_serde::to_vec_named(msg).unwrap();
        let len = bin.len() as u32;
        self.stdin.write_u32_le(len).await?;
        self.stdin.write_all(&bin).await?;
        Ok(())
    }

    async fn read_msg<T: DeserializeOwned>(&mut self) -> anyhow::Result<T> {
        let len = self.stdout.read_u32_le().await?;
        let mut buf = vec![0; len as usize];
        self.stdout.read_exact(&mut buf).await?;
        let res: Result<T, String> = rmp_serde::from_slice(&buf)?;
        match res {
            Ok(res) => Ok(res),
            Err(err) => {
                error!(%err, "Python subprocess returned an error.");
                Err(anyhow::anyhow!(err))
            }
        }
    }

    pub async fn init(
        &mut self,
        model_path: &str,
        dtype_str: &str,
        block_len: u32,
        kvpool_capacity: u32,
    ) -> anyhow::Result<()> {
        let init = Request::Init(Init {
            use_fake: false,
            model_path,
            dtype_str,
            block_len,
            kvpool_capacity,
        });
        self.write_msg(&init).await?;
        self.read_msg::<i32>().await?;
        Ok(())
    }

    pub async fn init_fake(&mut self) -> anyhow::Result<()> {
        let init = Request::Init(Init {
            use_fake: true,
            model_path: "",
            dtype_str: "",
            block_len: 0,
            kvpool_capacity: 0,
        });
        self.write_msg(&init).await?;
        self.read_msg::<i32>().await?;
        Ok(())
    }

    pub async fn shutdown(&mut self) -> anyhow::Result<()> {
        let shutdown = Request::Shutdown(Shutdown {});
        self.write_msg(&shutdown).await
    }

    pub async fn add_request(
        &mut self,
        reqid: Uuid,
        input_ids: &[u32],
        gencfg: &comm::GenerationConfig,
    ) -> anyhow::Result<()> {
        let add_request =
            Request::AddRequest(AddRequest { reqid, input_ids, gencfg });
        self.write_msg(&add_request).await?;
        self.read_msg::<i32>().await?;
        Ok(())
    }

    pub async fn cancel_request(&mut self, reqid: Uuid) -> anyhow::Result<()> {
        let cancel_request = Request::CancelRequest(CancelRequest { reqid });
        self.write_msg(&cancel_request).await?;
        self.read_msg::<i32>().await?;
        Ok(())
    }

    pub async fn batch_prefill(
        &mut self,
        reqids: &[Uuid],
    ) -> anyhow::Result<TextGenerationChunkResponse> {
        let batch_prefill = Request::BatchPrefill(BatchPrefill { reqids });
        self.write_msg(&batch_prefill).await?;
        self.read_msg().await
    }

    pub async fn batch_decode(
        &mut self,
        reqids: &[Uuid],
    ) -> anyhow::Result<TextGenerationChunkResponse> {
        let batch_decode = Request::BatchDecode(BatchDecode { reqids });
        self.write_msg(&batch_decode).await?;
        self.read_msg().await
    }
}

struct UnfinishedRequests {
    cancel: HashSet<Uuid>,
    prefill: Vec<(Uuid, Vec<u32>, comm::GenerationConfig)>,
    decode: Vec<Uuid>,
}

pub struct GpuExecutor {
    unfinished: Arc<Mutex<UnfinishedRequests>>,
    state: State,
}

enum State {
    Invalid,
    Spawned(SpawnedState),
    Running(BackToBackScheduleHandle),
}

struct SpawnedState {
    subprocess: ExecutorSubprocess,
    tx: mpsc::UnboundedSender<comm::TextGenChunk>,
}

struct BackToBackScheduleHandle {
    notify_new_task: Arc<Notify>,
    notify_shutdown: Arc<Notify>,
    join: JoinHandle<()>,
}

struct BackToBackSchedule {
    new_task: Arc<Notify>,
    shutdown: Arc<Notify>,
    subprocess: ExecutorSubprocess,
    unfinished: Arc<Mutex<UnfinishedRequests>>,
    tx: mpsc::UnboundedSender<comm::TextGenChunk>,
}

impl GpuExecutor {
    pub fn spawn(
        gpu_uuid: Uuid,
    ) -> anyhow::Result<(
        Child,
        mpsc::UnboundedReceiver<comm::TextGenChunk>,
        Self,
    )> {
        let (child, subprocess) = ExecutorSubprocess::spawn(gpu_uuid)?;
        let unfinished = Arc::new(Mutex::new(UnfinishedRequests {
            cancel: HashSet::new(),
            prefill: Vec::new(),
            decode: Vec::new(),
        }));
        let (tx, rx) = mpsc::unbounded_channel();
        let state = State::Spawned(SpawnedState { subprocess, tx });

        Ok((child, rx, Self { unfinished, state }))
    }

    fn unwrap_spawned_mut(&mut self) -> &mut SpawnedState {
        match &mut self.state {
            State::Spawned(s) => s,
            _ => panic!("Not in the Spawned state"),
        }
    }

    fn unwrap_bh_mut(&mut self) -> &mut BackToBackScheduleHandle {
        match &mut self.state {
            State::Running(bh) => bh,
            _ => panic!("Not in the Running state"),
        }
    }

    fn start_back_to_back_schedule(&mut self) {
        let state = std::mem::replace(&mut self.state, State::Invalid);
        match state {
            State::Spawned(SpawnedState { subprocess, tx }) => {
                let notify_new_task = Arc::new(Notify::new());
                let notify_shutdown = Arc::new(Notify::new());
                let mut b = BackToBackSchedule {
                    new_task: notify_new_task.clone(),
                    shutdown: notify_shutdown.clone(),
                    subprocess,
                    unfinished: self.unfinished.clone(),
                    tx,
                };
                let join = tokio::spawn(async move { b.run().await });
                let bh = BackToBackScheduleHandle {
                    notify_new_task,
                    notify_shutdown,
                    join,
                };
                self.state = State::Running(bh);
            }
            _ => panic!("Bad state"),
        };
    }

    pub async fn init(
        &mut self,
        model_path: &str,
        dtype_str: &str,
        block_len: u32,
        kvpool_capacity: u32,
    ) -> anyhow::Result<()> {
        self.unwrap_spawned_mut()
            .subprocess
            .init(model_path, dtype_str, block_len, kvpool_capacity)
            .await?;
        self.start_back_to_back_schedule();
        Ok(())
    }

    pub async fn init_fake(&mut self) -> anyhow::Result<()> {
        self.unwrap_spawned_mut().subprocess.init_fake().await?;
        self.start_back_to_back_schedule();
        Ok(())
    }

    pub async fn shutdown(&mut self) {
        let state = std::mem::replace(&mut self.state, State::Invalid);
        match state {
            State::Running(bh) => {
                bh.notify_shutdown.notify_one();
                bh.join.await.unwrap();
            }
            _ => panic!("Bad state"),
        }
    }

    pub async fn add_request(
        &mut self,
        reqid: Uuid,
        input_ids: Vec<u32>,
        gencfg: comm::GenerationConfig,
    ) {
        self.unfinished.lock().await.prefill.push((reqid, input_ids, gencfg));
        self.unwrap_bh_mut().notify_new_task.notify_one();
    }

    pub async fn cancel_request(&mut self, reqid: Uuid) {
        self.unfinished.lock().await.cancel.insert(reqid);
        self.unwrap_bh_mut().notify_new_task.notify_one();
    }
}

impl BackToBackSchedule {
    async fn run(&mut self) {
        loop {
            match self.process_unfinished().await {
                Err(_) => break,
                Ok(true) => continue,
                Ok(false) => {
                    tokio::select! {
                        _ = self.new_task.notified() => continue,
                        _ = self.shutdown.notified() => {
                            self.subprocess.shutdown().await.unwrap();
                            break;
                        },
                    }
                }
            }
        }
    }

    async fn process_unfinished(&mut self) -> anyhow::Result<bool> {
        enum Task {
            Cancel(HashSet<Uuid>),
            Prefill(Vec<(Uuid, Vec<u32>, comm::GenerationConfig)>),
            Decode(Vec<Uuid>),
            None,
        }
        let task = {
            let mut unfinished = self.unfinished.lock().await;
            if !unfinished.cancel.is_empty() {
                Task::Cancel(unfinished.cancel.drain().collect())
            } else if !unfinished.prefill.is_empty() {
                Task::Prefill(unfinished.prefill.drain(..).collect())
            } else if !unfinished.decode.is_empty() {
                Task::Decode(unfinished.decode.drain(..).collect())
            } else {
                Task::None
            }
        };
        match task {
            Task::Cancel(cancel) => {
                self.process_unfinished_cancel(cancel).await?;
            }
            Task::Prefill(prefill) => {
                self.process_unfinished_prefill(prefill).await?;
            }
            Task::Decode(decode) => {
                self.process_unfinished_decode(decode).await?;
            }
            Task::None => return Ok(false),
        }
        Ok(true)
    }

    async fn process_unfinished_cancel(
        &mut self,
        mut cancel: HashSet<Uuid>,
    ) -> anyhow::Result<()> {
        {
            let mut unfinished = self.unfinished.lock().await;
            let mut i = 0usize;
            while i < unfinished.prefill.len() && !cancel.is_empty() {
                let reqid = unfinished.prefill[i].0;
                if cancel.contains(&reqid) {
                    unfinished.prefill.swap_remove(i);
                    cancel.remove(&reqid);
                } else {
                    i += 1;
                }
            }
        }

        for reqid in &cancel {
            self.subprocess.cancel_request(*reqid).await?;
        }
        let mut unfinished = self.unfinished.lock().await;
        unfinished.decode.retain(|reqid| !cancel.contains(reqid));
        Ok(())
    }

    async fn process_unfinished_prefill(
        &mut self,
        prefill: Vec<(Uuid, Vec<u32>, comm::GenerationConfig)>,
    ) -> anyhow::Result<()> {
        let mut reqids = Vec::with_capacity(prefill.len());
        for (reqid, input_ids, gencfg) in &prefill {
            self.subprocess.add_request(*reqid, input_ids, gencfg).await?;
            reqids.push(*reqid);
        }
        let res = self.subprocess.batch_prefill(&reqids).await?;
        let mut remain = self.process_chunks(reqids, res)?;
        self.unfinished.lock().await.decode.append(&mut remain);
        Ok(())
    }

    async fn process_unfinished_decode(
        &mut self,
        decode: Vec<Uuid>,
    ) -> anyhow::Result<()> {
        let res = self.subprocess.batch_decode(&decode).await?;
        let mut remain = self.process_chunks(decode, res)?;
        self.unfinished.lock().await.decode.append(&mut remain);
        Ok(())
    }

    fn process_chunks(
        &self,
        reqs: Vec<Uuid>,
        res: TextGenerationChunkResponse,
    ) -> anyhow::Result<Vec<Uuid>> {
        let mut remain = Vec::with_capacity(reqs.len());
        for (reqid, token_id, finish) in
            izip!(reqs, res.token_ids, res.finish_reasons)
        {
            if finish == comm::FinishReason::NotFinished {
                remain.push(reqid);
            }
            self.tx.send(comm::TextGenChunk {
                request_id: reqid,
                token_id,
                finish_reason: finish,
            })?;
        }
        Ok(remain)
    }
}
