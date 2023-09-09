use std::net::SocketAddr;
use std::ops::ControlFlow;
use std::sync::Arc;

use axum::extract::ws::{Message, WebSocket};
use axum::extract::{ConnectInfo, Path, State, WebSocketUpgrade};
use axum::response::IntoResponse;
use axum::routing::get;
use axum::Router;
use futures::stream::{SplitSink, SplitStream};
use futures::{SinkExt, StreamExt};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tower::ServiceBuilder;
use tower_http::trace::{DefaultMakeSpan, TraceLayer};
use tower_http::ServiceBuilderExt;

use super::Scheduler;
use crate::comm;

pub async fn run_http(
    bind: SocketAddr,
    http_handle: axum_server::Handle,
    shutdown: CancellationToken,
    shutdown_complete: mpsc::Sender<()>,
    scheduler: Arc<Scheduler>,
) {
    let service = ServiceBuilder::new().catch_panic().layer(
        TraceLayer::new_for_http()
            .make_span_with(DefaultMakeSpan::default().include_headers(true)),
    );
    let ctx = Arc::new(HttpContext { shutdown, shutdown_complete, scheduler });
    let app = Router::new()
        .route("/rpc/:version/:node_type", get(ws_handler))
        .layer(service)
        .with_state(ctx);

    let server = axum_server::bind(bind);
    info!("Started HTTP server on {}", bind);
    let ret = server
        .handle(http_handle)
        .serve(app.into_make_service_with_connect_info::<SocketAddr>())
        .await;
    if let Err(e) = ret {
        error!("HTTP server error: {}", e);
    }
    info!("HTTP server stopped");
}

struct HttpContext {
    shutdown: CancellationToken,
    shutdown_complete: mpsc::Sender<()>,
    scheduler: Arc<Scheduler>,
}

async fn ws_handler(
    Path((version, node_type)): Path<(String, String)>,
    ws: WebSocketUpgrade,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    State(ctx): State<Arc<HttpContext>>,
) -> impl IntoResponse {
    if version != "v1" {
        return (axum::http::StatusCode::BAD_REQUEST, "Bad version")
            .into_response();
    }
    let node_type = match node_type.as_str() {
        "runner" => comm::NodeType::Runner,
        "apisrv" => comm::NodeType::ApiServer,
        "scheduler" => comm::NodeType::Scheduler,
        _ => {
            return (axum::http::StatusCode::BAD_REQUEST, "Bad node type")
                .into_response();
        }
    };
    ws.on_upgrade(move |socket| handle_socket(socket, addr, node_type, ctx))
}

async fn handle_socket(
    ws: WebSocket,
    addr: SocketAddr,
    node_type: comm::NodeType,
    ctx: Arc<HttpContext>,
) {
    tokio::spawn(Connection::serve(
        addr,
        ws,
        ctx.shutdown.clone(),
        ctx.shutdown_complete.clone(),
        ctx.scheduler.clone(),
        node_type,
    ));
}

struct Connection {
    addr: SocketAddr,
    ws_send: SplitSink<WebSocket, Message>,
    ws_recv: SplitStream<WebSocket>,
    ch_send: mpsc::Sender<Message>,
    ch_recv: mpsc::Receiver<Message>,
    shutdown: CancellationToken,
    _shutdown_complete: mpsc::Sender<()>,
    scheduler: Arc<Scheduler>,
    node_type: comm::NodeType,
}

impl Connection {
    pub async fn serve(
        addr: SocketAddr,
        ws: WebSocket,
        shutdown: CancellationToken,
        shutdown_complete: mpsc::Sender<()>,
        scheduler: Arc<Scheduler>,
        node_type: comm::NodeType,
    ) {
        let (ws_send, ws_recv) = ws.split();
        let (ch_send, ch_recv) = mpsc::channel(32);
        let mut conn = Self {
            addr,
            ws_send,
            ws_recv,
            ch_send,
            ch_recv,
            shutdown,
            _shutdown_complete: shutdown_complete,
            node_type,
            scheduler,
        };

        match conn.run().await {
            Ok(()) => info!(%conn.addr, ?conn.node_type, "Connection closed."),
            Err(e) => {
                error!(%conn.addr, ?conn.node_type, cause=%e, "Connection error. Connection closed.");
            }
        }
    }

    async fn run(&mut self) -> anyhow::Result<()> {
        while !self.shutdown.is_cancelled() {
            let ret = tokio::select! {
                msg = self.ws_recv.next() => {
                    self.handle_message(msg).await?
                },
                recv = self.ch_recv.recv() => {
                    self.forward_message(recv).await?
                },
                _ = self.shutdown.cancelled() => ControlFlow::Break(()),
            };
            match ret {
                ControlFlow::Continue(()) => (),
                ControlFlow::Break(()) => break,
            }
        }
        Ok(())
    }

    async fn handle_message(
        &mut self,
        msg: Option<Result<Message, axum::Error>>,
    ) -> anyhow::Result<ControlFlow<()>> {
        match msg {
            Some(Ok(Message::Text(_))) => {
                Err(anyhow::anyhow!("Unexpected text message"))
            }
            Some(Ok(Message::Ping(_))) => {
                // Nothing to do. Handled by websocket library.
                Ok(ControlFlow::Continue(()))
            }
            Some(Ok(Message::Pong(_))) => {
                // Nothing to do.
                Ok(ControlFlow::Continue(()))
            }
            Some(Ok(Message::Close(_))) => Ok(ControlFlow::Break(())),
            Some(Err(e)) => Err(e.into()),
            None => Ok(ControlFlow::Continue(())),
            Some(Ok(Message::Binary(m))) => {
                match self.node_type {
                    comm::NodeType::Runner => {
                        self.handle_runner_message(postcard::from_bytes(&m)?)
                            .await?;
                    }
                    comm::NodeType::ApiServer => todo!(),
                    comm::NodeType::Scheduler => todo!(),
                }
                Ok(ControlFlow::Continue(()))
            }
        }
    }

    async fn forward_message(
        &mut self,
        recv: Option<Message>,
    ) -> anyhow::Result<ControlFlow<()>> {
        match recv {
            Some(msg) => {
                self.ws_send.send(msg).await?;
                Ok(ControlFlow::Continue(()))
            }
            None => Ok(ControlFlow::Break(())),
        }
    }

    async fn handle_runner_message(
        &mut self,
        msg: comm::RunnerToSchedulerMessage,
    ) -> anyhow::Result<()> {
        use comm::RunnerToSchedulerMessage as M;
        match msg {
            M::AddRunnerRequest(m) => {
                self.scheduler
                    .add_runner(self.addr, self.ch_send.clone(), m)
                    .await
            }
            M::DelRunnerRequest(_) => todo!(),
            M::AcquireGpuResponse(m) => {
                self.scheduler.handle_acquire_gpu_resp(m)
            }
            M::ReleaseGpuResponse(_) => todo!(),
            M::RunnerMigrateToNewSchedulerCommand(_) => todo!(),
            M::RunnerMigratedToNewScheduler(_) => todo!(),
            M::TextGenChunk(_) => todo!(),
            M::UpdateGpuStatsRequest(_) => todo!(),
        }

        Ok(())
    }
}
