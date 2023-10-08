use std::path::PathBuf;

use dashmap::DashMap;
use tokio::sync::mpsc;
use tokio_tungstenite::tungstenite::Message;
use uuid::Uuid;

use super::executor::GpuExecutor;
use crate::comm;

struct Gpu {
    devprop: comm::CudaDeviceProp,
    rx: Option<mpsc::UnboundedReceiver<comm::BatchedTextGenChunk>>,
    executor: GpuExecutor,
}

pub struct Runner {
    model_path: PathBuf,
    use_fake_executor: bool,
    limit_gpumem: Option<u64>,

    uuid: Uuid,
    gpus: DashMap<Uuid, Gpu>,
    scheduler_tx: mpsc::Sender<Message>,
}

impl Runner {
    pub fn new(
        model_path: PathBuf,
        use_fake_executor: bool,
        limit_gpumem_gb: Option<u64>,
        devprops: Vec<comm::CudaDeviceProp>,
        gpu_executors: Vec<(
            mpsc::UnboundedReceiver<comm::BatchedTextGenChunk>,
            GpuExecutor,
        )>,
        scheduler_tx: mpsc::Sender<Message>,
    ) -> anyhow::Result<Self> {
        let uuid = Uuid::now_v7();
        let gpus = DashMap::new();
        for (devprop, (rx, executor)) in devprops.into_iter().zip(gpu_executors)
        {
            gpus.insert(devprop.uuid, Gpu { devprop, rx: Some(rx), executor });
        }
        Ok(Self {
            model_path,
            use_fake_executor,
            limit_gpumem: limit_gpumem_gb.map(|gb| gb * 1024 * 1024 * 1024),
            uuid,
            gpus,
            scheduler_tx,
        })
    }

    async fn send_message(&self, msg: &comm::RunnerToSchedulerMessage) {
        let binary =
            postcard::to_stdvec(msg).expect("Failed to serialize message");
        self.scheduler_tx.send(Message::Binary(binary)).await.unwrap();
    }

    pub async fn register_to_scheduler(&self) {
        self.send_message(&comm::RunnerToSchedulerMessage::AddRunnerRequest(
            comm::AddRunnerRequest {
                runner_id: self.uuid,
                devices: self
                    .gpus
                    .iter()
                    .map(|gpu| gpu.devprop.clone())
                    .collect(),
                limit_gpumem: self.limit_gpumem,
            },
        ))
        .await;
    }

    pub async fn acquire_gpu(
        &self,
        msg: &comm::AcquireGpuCommand,
    ) -> anyhow::Result<()> {
        let mut gpu = self.gpus.get_mut(&msg.gpu_uuid).unwrap();
        if self.use_fake_executor {
            gpu.executor.init_fake().await?;
        } else {
            gpu.executor
                .init(
                    &self.model_path.display().to_string(),
                    &msg.dtype,
                    msg.block_len,
                    msg.kvpool_capacity,
                )
                .await?;
        }

        let mut gpu_rx = gpu.rx.take().unwrap();
        let scheduler_tx = self.scheduler_tx.clone();
        tokio::spawn(async move {
            while let Some(chunk) = gpu_rx.recv().await {
                let msg =
                    comm::RunnerToSchedulerMessage::BatchedTextGenChunk(chunk);
                scheduler_tx
                    .send(Message::Binary(postcard::to_stdvec(&msg).unwrap()))
                    .await
                    .unwrap();
            }
        });

        self.send_message(&comm::RunnerToSchedulerMessage::AcquireGpuResponse(
            comm::AcquireGpuResponse { gpu_uuid: msg.gpu_uuid },
        ))
        .await;
        info!(
            gpu_uuid = %gpu.devprop.uuid,
            gpu_name = %gpu.devprop.name,
            block_len = %msg.block_len,
            kvpool_capacity = %msg.kvpool_capacity,
            "Initialized GpuExecutor.",
        );
        Ok(())
    }

    pub async fn release_gpu(
        &self,
        msg: &comm::ReleaseGpuCommand,
    ) -> anyhow::Result<()> {
        let mut gpu = self.gpus.get_mut(&msg.gpu_uuid).unwrap();
        gpu.executor.shutdown().await;
        self.send_message(&comm::RunnerToSchedulerMessage::ReleaseGpuResponse(
            comm::ReleaseGpuResponse { gpu_uuid: msg.gpu_uuid },
        ))
        .await;
        info!(
            gpu_uuid = %gpu.devprop.uuid,
            gpu_name = %gpu.devprop.name,
            "Shutdown GpuExecutor.",
        );
        Ok(())
    }

    pub async fn run_textgen(
        &self,
        msg: comm::RunTextGenCommand,
    ) -> anyhow::Result<()> {
        let mut gpu = self.gpus.get_mut(&msg.gpu_uuid).unwrap();
        gpu.executor
            .add_request(msg.req.request_id, msg.req.input_ids, msg.req.gencfg)
            .await;

        Ok(())
    }

    pub async fn cancel_textgen(
        &self,
        msg: &comm::CancelTextGenCommand,
    ) -> anyhow::Result<()> {
        let mut gpu = self.gpus.get_mut(&msg.gpu_uuid).unwrap();
        gpu.executor.cancel_request(msg.request_id).await;
        Ok(())
    }

    pub async fn handle_scheduler_message(
        &self,
        msg: comm::SchedulerToRunnerMessage,
    ) -> anyhow::Result<()> {
        use comm::SchedulerToRunnerMessage::*;
        match msg {
            AcquireGpuCommand(msg) => {
                self.acquire_gpu(&msg).await?;
            }
            ReleaseGpuCommand(msg) => {
                self.release_gpu(&msg).await?;
            }
            RunTextGenCommand(msg) => {
                self.run_textgen(msg).await?;
            }
            CancelTextGen(msg) => {
                self.cancel_textgen(&msg).await?;
            }
        }
        Ok(())
    }
}
