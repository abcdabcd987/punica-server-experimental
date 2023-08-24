use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use axum::extract::ws::WebSocket;
use axum::extract::WebSocketUpgrade;
use axum::response::IntoResponse;
use axum::routing::get;
use axum::Router;
use tokio::signal::unix::{signal, SignalKind};
use tokio::sync::Notify;
use tower::ServiceBuilder;
use tower_http::trace::{DefaultMakeSpan, TraceLayer};
use tower_http::ServiceBuilderExt;

#[derive(Debug, clap::Args)]
pub struct SchedulerArgs {
    #[arg(long, default_value = "[::1]:23081")]
    pub ws_bind: SocketAddr,
}

pub async fn scheduler_main(args: SchedulerArgs) -> anyhow::Result<()> {
    let env = Env::new();
    tokio::spawn(run_http(env.clone(), args.ws_bind));
    setup_signals(env.clone()).await;
    Ok(())
}

struct Env {
    graceful: Arc<Notify>,
    shutdown: Arc<Notify>,
    http_handle: axum_server::Handle,
}

impl Env {
    fn new() -> Arc<Env> {
        Arc::new(Env {
            graceful: Arc::new(Notify::new()),
            shutdown: Arc::new(Notify::new()),
            http_handle: axum_server::Handle::new(),
        })
    }
}

async fn listen_signal() -> String {
    let mut sigint = signal(SignalKind::interrupt()).unwrap();
    let mut sigquit = signal(SignalKind::quit()).unwrap();
    let mut sigterm = signal(SignalKind::terminate()).unwrap();
    tokio::select! {
        _ = sigint.recv() => String::from("SIGINT"),
        _ = sigquit.recv() => String::from("SIGQUIT"),
        _ = sigterm.recv() => String::from("SIGTERM"),
    }
}

async fn setup_signals(env: Arc<Env>) {
    let sig = listen_signal().await;
    info!(
        "Got {}. Starting graceful shutdown. A second signal will shutdown immediately.",
        sig
    );

    env.graceful.notify_waiters();
    let graceful = {
        let env = env.clone();
        tokio::spawn(async move {
            env.http_handle.graceful_shutdown(None);
            while env.http_handle.connection_count() > 0 {
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        })
    };

    let shutdown = async {
        let sig = listen_signal().await;
        info!("Got {}. Shutting down immediately.", sig);
        env.shutdown.notify_waiters();
        env.http_handle.shutdown();
    };

    tokio::select! {
        _ = graceful => (),
        _ = shutdown => (),
    }
}

async fn run_http(env: Arc<Env>, bind: SocketAddr) {
    let service = ServiceBuilder::new().catch_panic().layer(
        TraceLayer::new_for_http()
            .make_span_with(DefaultMakeSpan::default().include_headers(true)),
    );
    let app = Router::new().route("/rpc", get(ws_handler)).layer(service);

    let server = axum_server::bind(bind);
    info!("Started HTTP server on {}", bind);
    let ret = server
        .handle(env.http_handle.clone())
        .serve(app.into_make_service_with_connect_info::<SocketAddr>())
        .await;
    if let Err(e) = ret {
        error!("HTTP server error: {}", e);
    }
    info!("HTTP server stopped");
}

async fn ws_handler(ws: WebSocketUpgrade) -> impl IntoResponse {
    ws.on_upgrade(handle_socket)
}

async fn handle_socket(_ws: WebSocket) {
    println!("handle_socket");
}
