mod conn;
mod device_query;
mod tokenizer;

use std::path::PathBuf;

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use url::Url;
use uuid::Uuid;

use self::conn::{SchedulerConnection, SchedulerMessage};
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
    let runner = Runner::new()?;
    info!(devices = ?runner.devprops, "GPUs");

    let mut url = args.scheduler_url.clone();
    url.path_segments_mut().unwrap().push("v1").push("runner");

    let ct = CancellationToken::new();
    let mut conn = SchedulerConnection::connect_to_scheduler(url).await?;

    runner.register_to_scheduler(&mut conn).await?;

    let (shutdown_complete_tx, mut shutdown_complete_rx) = mpsc::channel(1);
    let mut conn = ConnHandler {
        runner,
        conn,
        shutdown: ct.clone(),
        _shutdown_complete: shutdown_complete_tx,
    };
    let conn = tokio::spawn(async move { conn.run().await });

    let ret: anyhow::Result<()> = tokio::select! {
        ret = conn => ret.unwrap(),
        _ = tokio::signal::ctrl_c() => {
            info!("Ctrl-C received.");
            Ok(())
        }
    };
    if let Err(e) = ret {
        error!(%e);
    }

    info!("Shutting down...");
    ct.cancel();
    let _ = shutdown_complete_rx.recv().await;
    info!("Shutdown complete.");
    Ok(())
}

pub struct Runner {
    uuid: Uuid,
    devprops: Vec<comm::CudaDeviceProp>,
}

impl Runner {
    pub fn new() -> anyhow::Result<Self> {
        let uuid = Uuid::now_v7();
        let devprops = device_query()?;
        Ok(Self { uuid, devprops })
    }

    pub async fn register_to_scheduler(
        &self,
        conn: &mut SchedulerConnection,
    ) -> anyhow::Result<()> {
        conn.send_message(comm::RunnerToSchedulerMessage::AddRunnerRequest(
            comm::AddRunnerRequest {
                runner_id: self.uuid,
                devices: self.devprops.clone(),
            },
        ))
        .await?;
        Ok(())
    }

    pub fn handle_scheduler_message(
        &self,
        _conn: &mut SchedulerConnection,
        msg: comm::SchedulerToRunnerMessage,
    ) -> anyhow::Result<()> {
        info!(?msg, "TODO: handle_scheduler_message");
        Ok(())
    }
}

struct ConnHandler {
    runner: Runner,
    conn: SchedulerConnection,
    shutdown: CancellationToken,
    _shutdown_complete: mpsc::Sender<()>,
}

impl ConnHandler {
    async fn run(&mut self) -> anyhow::Result<()> {
        while !self.shutdown.is_cancelled() {
            let msg = tokio::select! {
                msg = self.conn.recv_message() => msg?,
                _ = self.shutdown.cancelled() => break,
            };
            match msg {
                SchedulerMessage::Message(msg) => {
                    self.runner.handle_scheduler_message(&mut self.conn, msg)?
                }
                SchedulerMessage::Alive => {}
                SchedulerMessage::End => break,
            }
        }
        self.conn.close().await?;
        Ok(())
    }
}
