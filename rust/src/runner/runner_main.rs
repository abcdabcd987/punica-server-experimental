use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Context;
use tokio::sync::mpsc;
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;
use url::Url;

use super::conn::SchedulerConnection;
use super::device_query::device_query;
use super::runner::Runner;
use crate::runner::executor::GpuExecutor;

#[derive(Debug, clap::Args)]
pub struct RunnerArgs {
    #[arg(long)]
    pub model_path: PathBuf,
    #[arg(long, help = "wss://example.com/rpc")]
    pub scheduler_url: Url,
    #[arg(long, help = "Use FakeGpuExecutor")]
    pub fake_executor: bool,
}

pub async fn runner_main(args: RunnerArgs) -> anyhow::Result<()> {
    let ct = CancellationToken::new();
    let (shutdown_complete_tx, mut shutdown_complete_rx) = mpsc::channel(1);

    let mut url = args.scheduler_url.clone();
    url.path_segments_mut().unwrap().push("v1").push("runner");

    let (ws, _) = tokio_tungstenite::connect_async(&url)
        .await
        .with_context(|| format!("Failed to connect to scheduler: {}", url))?;

    let devprops = device_query()?;
    let mut gpu_executors = Vec::new();
    let mut wait_executors = JoinSet::new();
    for devprop in &devprops {
        let gpu_uuid = devprop.uuid;
        let (mut child, rx, executor) = GpuExecutor::spawn(gpu_uuid)?;
        wait_executors.spawn(async move { (gpu_uuid, child.wait().await) });
        gpu_executors.push((rx, executor));
    }

    let (ch_send, ch_recv) = mpsc::channel(32);
    let runner = Arc::new(Runner::new(
        args.model_path,
        args.fake_executor,
        devprops,
        gpu_executors,
        ch_send,
    )?);
    let conn = SchedulerConnection::new(
        url,
        ws,
        ch_recv,
        ct.clone(),
        shutdown_complete_tx.clone(),
        runner.clone(),
    );

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
    while let Some(res) = wait_executors.join_next().await {
        let (gpu_uuid, res) = res.unwrap();
        match res.map(|s| s.code()) {
            Ok(Some(0)) => {
                info!(gpu_uuid=%gpu_uuid, "GpuExecutor exited normally.")
            }
            Ok(Some(ec)) => {
                error!(gpu_uuid=%gpu_uuid, exit_code=%ec, "GpuExecutor exited with error.")
            }
            Ok(None) => {
                error!(gpu_uuid=%gpu_uuid, "GpuExecutor exited by signal.")
            }
            Err(e) => {
                error!(gpu_uuid=%gpu_uuid, cause=%e, "GpuExecutor exited with error.")
            }
        }
    }
    let _ = shutdown_complete_rx.recv().await;
    info!("Shutdown complete.");
    Ok(())
}
