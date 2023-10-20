use std::net::SocketAddr;
use std::ops::ControlFlow;
use std::sync::{Arc, Mutex};

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
use uuid::Uuid;

use super::scheduler::Scheduler;
use super::traits::{RequestStub, RunnerStub};
use crate::comm;

pub struct RunnerDelegate {
    runner_id: Uuid,
    addr: SocketAddr,
    devprops: Vec<comm::CudaDeviceProp>,
    limit_gpumem: Option<u64>,
    tx: mpsc::UnboundedSender<Message>,
}

impl RunnerStub for RunnerDelegate {
    fn id(&self) -> Uuid {
        self.runner_id
    }

    fn device_props(&self) -> &[comm::CudaDeviceProp] {
        &self.devprops
    }

    fn addr(&self) -> SocketAddr {
        self.addr
    }

    fn limit_gpumem(&self) -> Option<u64> {
        self.limit_gpumem
    }

    fn init_gpu(&self, msg: comm::AcquireGpuCommand) {
        self.send(comm::SchedulerToRunnerMessage::AcquireGpuCommand(msg));
    }

    fn run_textgen(&self, msg: comm::RunTextGenCommand) {
        self.send(comm::SchedulerToRunnerMessage::RunTextGenCommand(msg));
    }

    fn cancel_textgen(&self, msg: comm::CancelTextGenCommand) {
        self.send(comm::SchedulerToRunnerMessage::CancelTextGen(msg));
    }
}

impl RunnerDelegate {
    fn send(&self, msg: comm::SchedulerToRunnerMessage) {
        let m = postcard::to_allocvec(&msg).unwrap();
        let msg = Message::Binary(m);
        self.tx.send(msg).unwrap();
    }
}

pub struct Request {
    id: Uuid,
    lora_id: Uuid,
    prompt_len: u32,
    tokens: Vec<u32>,
    generation_config: comm::GenerationConfig,
    frontend_id: Uuid,
    tx: mpsc::UnboundedSender<Message>,
}

impl RequestStub for Request {
    fn id(&self) -> Uuid {
        self.id
    }

    fn lora_id(&self) -> Uuid {
        self.lora_id
    }

    fn input_ids(&self) -> &[u32] {
        &self.tokens[..self.prompt_len as usize]
    }

    fn generation_config(&self) -> &comm::GenerationConfig {
        &self.generation_config
    }

    fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    fn len(&self) -> u32 {
        self.tokens.len() as u32
    }

    fn frontend_id(&self) -> Uuid {
        self.frontend_id
    }

    fn add_chunk(&mut self, token_id: u32, finish: comm::FinishReason) {
        let index = self.tokens.len() as u32;
        self.tokens.push(token_id);

        let msg = comm::SchedulerToFrontendMessage::TextGenChunk(
            comm::TextGenChunk {
                request_id: self.id,
                index,
                token_id,
                finish_reason: finish,
            },
        );
        let m = postcard::to_allocvec(&msg).unwrap();
        let msg = Message::Binary(m);
        if let Err(e) = self.tx.send(msg) {
            error!(request_id=%self.id, cause=%e, "Failed to send message to frontend.");
        }
    }

    fn migrate(&mut self) {
        self.prompt_len = self.tokens.len() as u32;
    }
}

type SchedulerHandle = Arc<Mutex<Scheduler<RunnerDelegate, Request>>>;

pub async fn run_http(
    bind: SocketAddr,
    http_handle: axum_server::Handle,
    shutdown: CancellationToken,
    shutdown_complete: mpsc::Sender<()>,
    scheduler: SchedulerHandle,
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
    scheduler: SchedulerHandle,
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
    let conn_type = match node_type.as_str() {
        "runner" => {
            ConnectionType::Runner(RunnerConnection { runner_id: None })
        }
        "frontend" => ConnectionType::Frontend(FrontendConnection {
            frontend_id: Uuid::now_v7(),
        }),
        _ => {
            return (axum::http::StatusCode::BAD_REQUEST, "Bad node type")
                .into_response();
        }
    };
    ws.on_upgrade(move |socket| handle_socket(socket, addr, conn_type, ctx))
}

async fn handle_socket(
    ws: WebSocket,
    addr: SocketAddr,
    conn_type: ConnectionType,
    ctx: Arc<HttpContext>,
) {
    tokio::spawn(Connection::serve(
        addr,
        ws,
        ctx.shutdown.clone(),
        ctx.shutdown_complete.clone(),
        ctx.scheduler.clone(),
        conn_type,
    ));
}

enum ConnectionType {
    Runner(RunnerConnection),
    Frontend(FrontendConnection),
}

impl ConnectionType {
    pub fn type_name(&self) -> &'static str {
        match self {
            ConnectionType::Runner(_) => "runner",
            ConnectionType::Frontend(_) => "frontend",
        }
    }
}

struct RunnerConnection {
    runner_id: Option<Uuid>,
}

struct FrontendConnection {
    frontend_id: Uuid,
}

struct Connection {
    addr: SocketAddr,
    ws_send: SplitSink<WebSocket, Message>,
    ws_recv: SplitStream<WebSocket>,
    ch_send: mpsc::UnboundedSender<Message>,
    ch_recv: mpsc::UnboundedReceiver<Message>,
    shutdown: CancellationToken,
    _shutdown_complete: mpsc::Sender<()>,
    scheduler: SchedulerHandle,
    conn_type: ConnectionType,
}

impl Connection {
    pub async fn serve(
        addr: SocketAddr,
        ws: WebSocket,
        shutdown: CancellationToken,
        shutdown_complete: mpsc::Sender<()>,
        scheduler: SchedulerHandle,
        conn_type: ConnectionType,
    ) {
        let (ws_send, ws_recv) = ws.split();
        let (ch_send, ch_recv) = mpsc::unbounded_channel();
        let mut conn = Self {
            addr,
            ws_send,
            ws_recv,
            ch_send,
            ch_recv,
            shutdown,
            _shutdown_complete: shutdown_complete,
            conn_type,
            scheduler,
        };

        match conn.run().await {
            Ok(()) => {
                info!(addr=%conn.addr, conn_type=conn.conn_type.type_name(), "Connection closed.")
            }
            Err(e) => {
                error!(addr=%conn.addr, conn_type=conn.conn_type.type_name(), cause=%e, "Connection error. Connection closed.");
            }
        }
        match &conn.conn_type {
            ConnectionType::Runner(state) => {
                if let Some(runner_id) = state.runner_id {
                    conn.scheduler.lock().unwrap().del_runner(runner_id);
                }
            }
            ConnectionType::Frontend(state) => {
                conn.scheduler.lock().unwrap().del_frontend(state.frontend_id);
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
                match &mut self.conn_type {
                    ConnectionType::Runner(_) => {
                        self.handle_runner_message(postcard::from_bytes(&m)?)?;
                    }
                    ConnectionType::Frontend(_) => {
                        self.handle_frontend_message(postcard::from_bytes(
                            &m,
                        )?)?;
                    }
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

    fn handle_runner_message(
        &mut self,
        msg: comm::RunnerToSchedulerMessage,
    ) -> anyhow::Result<()> {
        use comm::RunnerToSchedulerMessage::*;
        match msg {
            AddRunnerRequest(m) => {
                let state = match &mut self.conn_type {
                    ConnectionType::Runner(state) => state,
                    _ => unreachable!(),
                };
                state.runner_id = Some(m.runner_id);

                let runner = RunnerDelegate {
                    runner_id: m.runner_id,
                    addr: self.addr,
                    devprops: m.devices,
                    limit_gpumem: m.limit_gpumem,
                    tx: self.ch_send.clone(),
                };
                self.scheduler.lock().unwrap().add_runner(runner);
            }
            AcquireGpuResponse(m) => {
                self.scheduler.lock().unwrap().notify_gpu_initialized(&m);
            }
            ReleaseGpuResponse(_) => todo!(),
            BatchedTextGenChunk(m) => {
                self.scheduler.lock().unwrap().notify_textgen_chunk(&m);
            }
        }

        Ok(())
    }

    fn handle_frontend_message(
        &mut self,
        msg: comm::FrontendToSchedulerMessage,
    ) -> anyhow::Result<()> {
        use comm::FrontendToSchedulerMessage::*;
        let state = match &mut self.conn_type {
            ConnectionType::Frontend(state) => state,
            _ => unreachable!(),
        };
        match msg {
            TextGenRequest(m) => {
                let req = Request {
                    id: m.request_id,
                    lora_id: m.lora_id,
                    prompt_len: m.input_ids.len() as u32,
                    tokens: m.input_ids,
                    generation_config: m.gencfg,
                    frontend_id: state.frontend_id,
                    tx: self.ch_send.clone(),
                };
                self.scheduler.lock().unwrap().add_textgen(req);
            }
            CancelTextGen(m) => {
                self.scheduler.lock().unwrap().cancel_textgen(&m);
            }
        }

        Ok(())
    }
}
