use std::net::SocketAddr;
use std::sync::{Arc, Mutex};

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use super::scheduler::Scheduler;
use super::server;
use crate::model_config;

#[derive(Debug, clap::Args)]
pub struct SchedulerArgs {
    #[arg(long, default_value = "127.0.0.1:23081")]
    pub ws_bind: SocketAddr,
}

pub async fn scheduler_main(args: SchedulerArgs) -> anyhow::Result<()> {
    let ct = CancellationToken::new();
    let http_handle = axum_server::Handle::new();
    let scheduler =
        Arc::new(Mutex::new(Scheduler::new(model_config::LLAMA_7B.clone())));
    let (shutdown_complete_tx, mut shutdown_complete_rx) = mpsc::channel(1);

    let http = tokio::spawn(server::run_http(
        args.ws_bind,
        http_handle.clone(),
        ct.clone(),
        shutdown_complete_tx.clone(),
        scheduler.clone(),
    ));
    drop(shutdown_complete_tx);

    let ret: anyhow::Result<()> = tokio::select! {
        ret = http => ret.map_err(|e| e.into()),
        _ = tokio::signal::ctrl_c() => {
            info!("Ctrl-C received.");
            Ok(())
        }
    };
    if let Err(e) = ret {
        error!(%e);
    }

    info!("Shutting down...");
    http_handle.graceful_shutdown(None);
    ct.cancel();
    // while http_handle.connection_count() > 0 {
    //     tokio::time::sleep(Duration::from_millis(100)).await;
    // }
    let _ = shutdown_complete_rx.recv().await;
    info!("Shutdown complete.");
    Ok(())
}
