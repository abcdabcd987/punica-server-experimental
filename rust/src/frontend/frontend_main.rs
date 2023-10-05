use std::net::SocketAddr;
use std::path::PathBuf;

use anyhow::Context;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use url::Url;

use super::api_server;
use crate::scheduler_client::SchedulerConnection;
use crate::tokenizer::Tokenizer;

#[derive(Debug, clap::Args)]
pub struct FrontendArgs {
    #[arg(long, default_value = "[::1]:23082")]
    pub bind: SocketAddr,
    #[arg(long)]
    pub model_path: PathBuf,
    #[arg(long, help = "wss://example.com/rpc")]
    pub scheduler_url: Url,
}

pub async fn frontend_main(args: FrontendArgs) -> anyhow::Result<()> {
    let ct = CancellationToken::new();
    let http_handle = axum_server::Handle::new();
    let (shutdown_complete_tx, mut shutdown_complete_rx) = mpsc::channel(1);

    let tokenizer = Tokenizer::new(&args.model_path).with_context(|| {
        format!("Failed to load tokenizer from {}", args.model_path.display())
    })?;

    let mut url = args.scheduler_url.clone();
    url.path_segments_mut().unwrap().push("v1").push("frontend");
    let (ws, _) = tokio_tungstenite::connect_async(&url)
        .await
        .with_context(|| format!("Failed to connect to scheduler: {}", url))?;

    let conn = SchedulerConnection::new(
        url,
        ws,
        ct.clone(),
        shutdown_complete_tx.clone(),
    );
    let scheduler_client = conn.get_client();
    let http = tokio::spawn(api_server::run_http(
        args.bind,
        http_handle.clone(),
        shutdown_complete_tx.clone(),
        scheduler_client,
        tokenizer,
    ));
    let conn = tokio::spawn(async move { conn.serve().await });

    let ret: anyhow::Result<()> = tokio::select! {
        ret = http => ret.map_err(|e| e.into()),
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
    http_handle.graceful_shutdown(None);
    let _ = shutdown_complete_rx.recv().await;
    info!("Shutdown complete.");

    Ok(())
}
