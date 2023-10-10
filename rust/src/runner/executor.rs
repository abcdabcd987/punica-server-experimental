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
use crate::paged_kv_tracker::PagedKvTracker;

#[derive(Serialize, Debug)]
#[serde(tag = "t", content = "c")]
enum Request<'a> {
    Init(Init<'a>),
    Shutdown(Shutdown),
    AddRequest(AddRequest<'a>),
    CancelRequest(CancelRequest),
    Step {},
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

#[derive(Deserialize, Debug)]
pub struct TextGenerationChunkResponse {
    pub request_ids: Vec<Uuid>,
    pub indicies: Vec<u32>,
    pub token_ids: Vec<u32>,
    pub finish_reasons: Vec<comm::FinishReason>,
    pub num_free_kv_blocks: u32,
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

    pub async fn step(
        &mut self,
    ) -> anyhow::Result<TextGenerationChunkResponse> {
        let step = Request::Step {};
        self.write_msg(&step).await?;
        self.read_msg().await
    }
}

// TODO: replace this with a command queue.
struct UnfinishedRequests {
    // Accessible by both the main thread and the back-to-back schedule thread.
    cancel: HashSet<Uuid>,
    enqueue: Vec<(Uuid, Vec<u32>, comm::GenerationConfig)>,

    // Only accessible by the back-to-back schedule thread.
    prefill: HashSet<Uuid>,
    decode: HashSet<Uuid>,
}

pub struct GpuExecutor {
    gpu_uuid: Uuid,
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
    tx: mpsc::UnboundedSender<comm::BatchedTextGenChunk>,
}

struct BackToBackScheduleHandle {
    notify_new_task: Arc<Notify>,
    notify_shutdown: Arc<Notify>,
    join: JoinHandle<()>,
}

struct BackToBackSchedule {
    gpu_uuid: Uuid,
    new_task: Arc<Notify>,
    shutdown: Arc<Notify>,
    subprocess: ExecutorSubprocess,
    unfinished: Arc<Mutex<UnfinishedRequests>>,
    tx: mpsc::UnboundedSender<comm::BatchedTextGenChunk>,
    kvpool: PagedKvTracker,
}

impl GpuExecutor {
    pub fn spawn(
        gpu_uuid: Uuid,
    ) -> anyhow::Result<(
        Child,
        mpsc::UnboundedReceiver<comm::BatchedTextGenChunk>,
        Self,
    )> {
        let (child, subprocess) = ExecutorSubprocess::spawn(gpu_uuid)?;
        let unfinished = Arc::new(Mutex::new(UnfinishedRequests {
            cancel: HashSet::new(),
            enqueue: Vec::new(),
            prefill: HashSet::new(),
            decode: HashSet::new(),
        }));
        let (tx, rx) = mpsc::unbounded_channel();
        let state = State::Spawned(SpawnedState { subprocess, tx });

        Ok((child, rx, Self { gpu_uuid, unfinished, state }))
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

    fn start_back_to_back_schedule(&mut self, kvpool: PagedKvTracker) {
        let state = std::mem::replace(&mut self.state, State::Invalid);
        match state {
            State::Spawned(SpawnedState { subprocess, tx }) => {
                let notify_new_task = Arc::new(Notify::new());
                let notify_shutdown = Arc::new(Notify::new());
                let mut b = BackToBackSchedule {
                    gpu_uuid: self.gpu_uuid,
                    new_task: notify_new_task.clone(),
                    shutdown: notify_shutdown.clone(),
                    subprocess,
                    unfinished: self.unfinished.clone(),
                    tx,
                    kvpool,
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
        let kvpool = PagedKvTracker::new(kvpool_capacity, block_len);
        self.start_back_to_back_schedule(kvpool);
        Ok(())
    }

    pub async fn init_fake(&mut self) -> anyhow::Result<()> {
        self.unwrap_spawned_mut().subprocess.init_fake().await?;
        let kvpool = PagedKvTracker::new(1000, 16);
        self.start_back_to_back_schedule(kvpool);
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
        self.unfinished.lock().await.enqueue.push((reqid, input_ids, gencfg));
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
        if self.process_unfinished_cancel().await? {
            return Ok(true);
        }
        if self.process_unfinished_enqueue().await? {
            return Ok(true);
        }
        if self.process_unfinished_step().await? {
            return Ok(true);
        }
        Ok(false)
    }

    async fn process_unfinished_cancel(&mut self) -> anyhow::Result<bool> {
        let mut unfinished_guard = self.unfinished.lock().await;
        let unfinished = &mut *unfinished_guard;
        if unfinished.cancel.is_empty() {
            return Ok(false);
        }

        // Remove from pending enqueue
        let mut i = 0usize;
        while i < unfinished.enqueue.len() && !unfinished.cancel.is_empty() {
            let reqid = unfinished.enqueue[i].0;
            if unfinished.cancel.contains(&reqid) {
                unfinished.enqueue.swap_remove(i);
                unfinished.cancel.remove(&reqid);
            } else {
                i += 1;
            }
        }

        // Remove from running requests
        for reqid in unfinished.cancel.iter() {
            self.subprocess.cancel_request(*reqid).await?;
            self.kvpool.release(*reqid);
            unfinished.prefill.remove(reqid);
            unfinished.decode.remove(reqid);
        }
        unfinished.cancel.clear();
        Ok(true)
    }

    async fn process_unfinished_enqueue(&mut self) -> anyhow::Result<bool> {
        let mut unfinished_guard = self.unfinished.lock().await;
        let unfinished = &mut *unfinished_guard;
        if unfinished.enqueue.is_empty() {
            return Ok(false);
        }

        // Don't enqueue if not enough free blocks
        let mut new_kv_blocks = 0;
        for (_reqid, input_ids, _gencfg) in &unfinished.enqueue {
            new_kv_blocks +=
                self.kvpool.calc_init_blocks(input_ids.len() as u32);
        }
        if new_kv_blocks > self.kvpool.num_free_blocks() {
            return Ok(false);
        }

        // Do enqueue
        let mut reqids = Vec::with_capacity(unfinished.enqueue.len());
        for (reqid, input_ids, gencfg) in &unfinished.enqueue {
            assert!(self.kvpool.init(*reqid, input_ids.len() as u32));
            self.subprocess.add_request(*reqid, input_ids, gencfg).await?;
            reqids.push(*reqid);
        }
        unfinished.prefill.extend(reqids);
        unfinished.enqueue.clear();
        Ok(true)
    }

    async fn process_unfinished_step(&mut self) -> anyhow::Result<bool> {
        let unfinished = self.unfinished.lock().await;
        if unfinished.prefill.is_empty() && unfinished.decode.is_empty() {
            return Ok(false);
        }

        // Don't step if not enough free blocks. Being conservative here.
        if self.kvpool.num_free_blocks() < unfinished.decode.len() as u32 {
            return Ok(false);
        }

        // Release lock. Otherwise it will block adding enqueue and cancel.
        let total_len = unfinished.prefill.len() + unfinished.decode.len();
        drop(unfinished);

        // Let the subprocess do its thing.
        let res: TextGenerationChunkResponse = self.subprocess.step().await?;

        // Reacquire lock and process the response.
        let mut unfinished_guard = self.unfinished.lock().await;
        let unfinished = &mut *unfinished_guard;

        // TODO: use a command queue to replace Arc<Mutex<UnfinishedRequests>>
        assert_eq!(
            total_len,
            unfinished.prefill.len() + unfinished.decode.len(),
            "Sanity check: nobody should touch unfinished.prefill and unfinished.decode while we don't hold the lock."
        );

        // Track KvCache usage: decode
        assert_eq!(total_len, res.request_ids.len());
        for reqid in unfinished.decode.iter() {
            assert!(self.kvpool.append_token(reqid));
        }

        // Move all prefill to decode
        unfinished.decode.extend(unfinished.prefill.drain());

        // Handle outputs
        let mut chunks = Vec::with_capacity(res.request_ids.len());
        for (reqid, index, token_id, finish) in izip!(
            res.request_ids,
            res.indicies,
            res.token_ids,
            res.finish_reasons
        ) {
            if finish != comm::FinishReason::NotFinished {
                assert!(unfinished.decode.remove(&reqid));

                // Track KvCache usage: finish
                self.kvpool.release(reqid);
            }
            chunks.push(comm::TextGenChunk {
                request_id: reqid,
                index,
                token_id,
                finish_reason: finish,
            });
        }

        // Track KvCache usage: free blocks
        assert_eq!(self.kvpool.num_free_blocks(), res.num_free_kv_blocks);

        self.tx.send(comm::BatchedTextGenChunk {
            chunks,
            gpu_uuid: self.gpu_uuid,
            num_free_kv_blocks: self.kvpool.num_free_blocks(),
        })?;

        Ok(true)
    }
}
