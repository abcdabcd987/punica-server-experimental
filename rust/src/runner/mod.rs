mod conn;
mod debug_executor_main;
mod device_query;
mod executor;
mod tokenizer;

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Context;
use tokio::sync::mpsc;
use tokio_tungstenite::tungstenite::Message;
use tokio_util::sync::CancellationToken;
use url::Url;
use uuid::Uuid;

use self::conn::SchedulerConnection;
pub use self::debug_executor_main::{debug_executor_main, DebugExecutorArgs};
use self::device_query::device_query;
use crate::comm;

#[derive(Debug, clap::Args)]
pub struct RunnerArgs {
    #[arg(long)]
    pub model_path: PathBuf,
    #[arg(long, help = "wss://example.com/rpc")]
    pub scheduler_url: Url,
}

pub async fn runner_main(args: RunnerArgs) -> anyhow::Result<()> {
    let ct = CancellationToken::new();
    let (shutdown_complete_tx, mut shutdown_complete_rx) = mpsc::channel(1);
    let runner = Arc::new(Runner::new()?);
    info!(devices = ?runner.devprops, "GPUs");

    let mut url = args.scheduler_url.clone();
    url.path_segments_mut().unwrap().push("v1").push("runner");

    let (ws, _) = tokio_tungstenite::connect_async(&url)
        .await
        .with_context(|| format!("Failed to connect to scheduler: {}", url))?;
    let conn = SchedulerConnection::new(
        url,
        ws,
        ct.clone(),
        shutdown_complete_tx.clone(),
        runner.clone(),
    );

    runner.set_scheduler_tx(conn.ch_send.clone()).await;
    runner.register_to_scheduler().await;

    let conn = tokio::spawn(async move { conn.serve().await });

    let ret: anyhow::Result<()> = tokio::select! {
        _ = conn => {
            Ok(())
        },
        _ = tokio::signal::ctrl_c() => {
            info!("Ctrl-C received.");
            Ok(())
        }
    };
    if let Err(e) = ret {
        error!(%e);
    }

    info!("Shutting down...");
    drop(shutdown_complete_tx);
    ct.cancel();
    let _ = shutdown_complete_rx.recv().await;
    info!("Shutdown complete.");
    Ok(())
}

pub struct Runner {
    uuid: Uuid,
    devprops: Vec<comm::CudaDeviceProp>,
    scheduler_tx: tokio::sync::RwLock<Option<mpsc::Sender<Message>>>,
}

impl Runner {
    pub fn new() -> anyhow::Result<Self> {
        let uuid = Uuid::now_v7();
        let devprops = device_query()?;
        Ok(Self {
            uuid,
            devprops,
            scheduler_tx: tokio::sync::RwLock::new(None),
        })
    }

    pub async fn set_scheduler_tx(&self, scheduler_tx: mpsc::Sender<Message>) {
        self.scheduler_tx.write().await.replace(scheduler_tx);
    }

    async fn send_message(&self, msg: &comm::RunnerToSchedulerMessage) {
        let binary =
            postcard::to_stdvec(msg).expect("Failed to serialize message");
        let scheduler_tx = self.scheduler_tx.read().await;
        scheduler_tx
            .as_ref()
            .unwrap()
            .send(Message::Binary(binary))
            .await
            .unwrap();
    }

    pub async fn register_to_scheduler(&self) {
        self.send_message(&comm::RunnerToSchedulerMessage::AddRunnerRequest(
            comm::AddRunnerRequest {
                runner_id: self.uuid,
                devices: self.devprops.clone(),
            },
        ))
        .await;
    }

    pub async fn handle_scheduler_message(
        &self,
        msg: comm::SchedulerToRunnerMessage,
    ) -> anyhow::Result<()> {
        info!(?msg, "TODO: handle_scheduler_message");
        Ok(())
    }
}
