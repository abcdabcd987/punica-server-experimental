use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;

use axum::extract::ws::{Message, WebSocket};
use axum::extract::{ConnectInfo, State, WebSocketUpgrade};
use axum::response::IntoResponse;
use axum::routing::get;
use axum::Router;
use futures::stream::{SplitSink, SplitStream};
use futures::{SinkExt, StreamExt};
use tokio::sync::mpsc::{Receiver, Sender};
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;
use tower::ServiceBuilder;
use tower_http::trace::{DefaultMakeSpan, TraceLayer};
use tower_http::ServiceBuilderExt;
use uuid::Uuid;

use crate::comm;

#[derive(Debug, clap::Args)]
pub struct SchedulerArgs {
    #[arg(long, default_value = "[::1]:23081")]
    pub ws_bind: SocketAddr,
}

pub async fn scheduler_main(args: SchedulerArgs) -> anyhow::Result<()> {
    let env = Env::new();
    tokio::select! {
        _ = tokio::spawn(run_http(env.clone(), args.ws_bind)) => (),
        _ = tokio::spawn(async {
            tokio::signal::ctrl_c().await.expect("Failed to setup signal handler");
            info!("Got ctrl-c. Starting graceful shutdown.");
        }) => (),
    }

    env.ct.cancel();
    env.http_handle.graceful_shutdown(None);
    while env.http_handle.connection_count() > 0 {
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    info!("Shutdown complete.");
    Ok(())
}

struct Env {
    ct: CancellationToken,
    http_handle: axum_server::Handle,
    scheduler: Arc<Scheduler>,
}

impl Env {
    fn new() -> Arc<Env> {
        Arc::new(Env {
            ct: CancellationToken::new(),
            http_handle: axum_server::Handle::new(),
            scheduler: Scheduler::new(),
        })
    }
}

async fn run_http(env: Arc<Env>, bind: SocketAddr) {
    let service = ServiceBuilder::new().catch_panic().layer(
        TraceLayer::new_for_http()
            .make_span_with(DefaultMakeSpan::default().include_headers(true)),
    );
    let app = Router::new()
        .route("/rpc", get(ws_handler))
        .layer(service)
        .with_state(env.clone());

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

#[derive(Debug, Clone, Copy)]
enum NodeType {
    Unknown,
    Runner,
    ApiServer,
    Scheduler,
}

impl std::fmt::Display for NodeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NodeType::Unknown => write!(f, "Unknown"),
            NodeType::Runner => write!(f, "Runner"),
            NodeType::ApiServer => write!(f, "ApiServer"),
            NodeType::Scheduler => write!(f, "Scheduler"),
        }
    }
}

struct Connection {
    addr: SocketAddr,
    ct: CancellationToken,
    node_type: RwLock<NodeType>,
}

impl Connection {
    fn new(addr: SocketAddr, ct: CancellationToken) -> Arc<Connection> {
        Arc::new(Connection {
            addr,
            ct,
            node_type: RwLock::new(NodeType::Unknown),
        })
    }
}

async fn ws_handler(
    ws: WebSocketUpgrade,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    State(env): State<Arc<Env>>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(env, socket, addr))
}

async fn handle_socket(env: Arc<Env>, ws: WebSocket, addr: SocketAddr) {
    let conn = Connection::new(addr, env.ct.child_token());

    serve_conn(env, conn, ws).await;
}

async fn serve_conn(env: Arc<Env>, conn: Arc<Connection>, ws: WebSocket) {
    let (ws_tx, ws_rx) = ws.split();
    let (ch_tx, ch_rx) = tokio::sync::mpsc::channel::<Message>(32);

    let mut tasks = JoinSet::new();
    tasks.spawn(conn_ping(conn.clone(), ch_tx.clone()));
    tasks.spawn(conn_write(conn.clone(), ws_tx, ch_rx));
    tasks.spawn(conn_read(env, conn.clone(), ws_rx, ch_tx));
    while let Some(res) = tasks.join_next().await {
        if let Err(e) = res {
            error!(
                "Error while handling RPC message. Peer: {}. Error: {}",
                conn.addr, e
            );
        }
    }
}

async fn conn_ping(conn: Arc<Connection>, tx: Sender<Message>) {
    let mut interval = tokio::time::interval(Duration::from_secs(5));
    loop {
        tokio::select! {
            _ = interval.tick() => {
                let msg = Message::Ping(vec![]);

                if let Err(e) = tx.send(msg).await {
                    error!("Failed to send ping to {}, closing conection. Error: {}", conn.addr, e);
                    conn.ct.cancel();
                    break;
                }
            },
            _ = conn.ct.cancelled() => break,
        }
    }
}

async fn conn_write(
    conn: Arc<Connection>,
    mut ws_tx: SplitSink<WebSocket, Message>,
    mut ch_rx: Receiver<Message>,
) {
    loop {
        tokio::select! {
            msg = ch_rx.recv() => {
                if let Some(msg) = msg {
                    if let Err(e) = ws_tx.send(msg).await {
                        error!("Failed to send message to {}, closing connection. Error: {}", conn.addr, e);
                        conn.ct.cancel();
                        break;
                    }
                }
            },
            _ = conn.ct.cancelled() => break,
        }
    }
}

async fn conn_read(
    env: Arc<Env>,
    conn: Arc<Connection>,
    mut ws_rx: SplitStream<WebSocket>,
    ch_tx: Sender<Message>,
) {
    let mut tasks = JoinSet::<()>::new();
    loop {
        tokio::select! {
            msg = ws_rx.next() => {
                match msg {
                    Some(Ok(Message::Ping(_))) => {
                        conn_update_liveness(conn.clone());
                        let tx = ch_tx.clone();
                        tasks.spawn(async move {
                            let _ =tx.send(Message::Pong(vec![])).await;
                        });
                    },
                    Some(Ok(Message::Pong(_))) => {
                        conn_update_liveness(conn.clone());
                    },
                    Some(Ok(Message::Binary(v))) => {
                        tasks.spawn({
                            let env = env.clone();
                            let conn = conn.clone();
                            let ch_tx = ch_tx.clone();
                            async move {
                                if let Err(e) = conn_handle_binary_message(env, conn.clone(), v, ch_tx).await {
                                    error!("Failed to parse message from {}, closing connection. Error: {}", conn.addr, e);
                                    conn.ct.cancel();
                                }
                            }
                        });
                    },
                    Some(Ok(Message::Text(_))) => {
                        error!("Unexpected text from {}, closing connection.", conn.addr);
                        conn.ct.cancel();
                        break;
                    },
                    Some(Ok(Message::Close(_))) => {
                        info!("Received close from {}", conn.addr);
                        conn.ct.cancel();
                        break;
                    },
                    Some(Err(e)) => {
                        error!("Failed to read message from {}, closing connection. Error: {}", conn.addr, e);
                        conn.ct.cancel();
                        break;
                    },
                    None =>break,
                }
            },
            _ = conn.ct.cancelled() => break,
        }
    }

    while let Some(res) = tasks.join_next().await {
        if let Err(e) = res {
            error!(
                "Error while handling RPC message. Peer: {}. Error: {}",
                conn.addr, e
            );
        }
    }
}

fn conn_update_liveness(_conn: Arc<Connection>) {
    // TODO
}

async fn conn_handle_binary_message(
    env: Arc<Env>,
    conn: Arc<Connection>,
    binary: Vec<u8>,
    tx: Sender<Message>,
) -> postcard::Result<()> {
    let node_type = *conn.node_type.read().unwrap();
    match node_type {
        NodeType::Unknown => {
            let msg = postcard::from_bytes::<comm::HelloScheduler>(&binary)?;
            let mut node_type = conn.node_type.write().unwrap();
            *node_type = match msg.node_type {
                comm::NodeType::Runner => NodeType::Runner,
                comm::NodeType::ApiServer => NodeType::ApiServer,
                comm::NodeType::Scheduler => NodeType::Scheduler,
            };
            info!(
                "New connection from {} registered as {}",
                conn.addr, node_type
            );
        }
        NodeType::Runner => {
            conn_handle_runner_message(
                env,
                conn,
                postcard::from_bytes(&binary)?,
                tx,
            )
            .await
        }
        NodeType::ApiServer => {
            conn_handle_apisrv_message(conn, postcard::from_bytes(&binary)?, tx)
                .await
        }
        NodeType::Scheduler => {
            conn_handle_scheduler_message(
                conn,
                postcard::from_bytes(&binary)?,
                tx,
            )
            .await
        }
    }
    Ok(())
}

async fn conn_handle_runner_message(
    env: Arc<Env>,
    conn: Arc<Connection>,
    msg: comm::RunnerToSchedulerMessage,
    tx: Sender<Message>,
) {
    use comm::RunnerToSchedulerMessage as M;
    debug!("{:?}", msg);
    match msg {
        M::AddRunnerRequest(m) => {
            env.scheduler.add_runner(conn.addr, tx, m).await
        }
        M::DelRunnerRequest(_) => todo!(),
        M::AcquireGpuResponse(m) => env.scheduler.acquire_gpu_resp(m),
        M::ReleaseGpuResponse(_) => todo!(),
        M::RunnerMigrateToNewSchedulerCommand(_) => todo!(),
        M::RunnerMigratedToNewScheduler(_) => todo!(),
        M::TextGenChunk(_) => todo!(),
        M::UpdateGpuStatsRequest(_) => todo!(),
    }
}

async fn conn_handle_apisrv_message(
    _conn: Arc<Connection>,
    _msg: comm::ApiServerToSchedulerMessage,
    _tx: Sender<Message>,
) {
    todo!();
}

async fn conn_handle_scheduler_message(
    _conn: Arc<Connection>,
    _msg: comm::SchedulerToSchedulerMessage,
    _tx: Sender<Message>,
) {
    todo!();
}

struct Scheduler {
    gpus: RwLock<HashMap<Uuid, Arc<Gpu>>>,
    runners: RwLock<HashMap<Uuid, Arc<Runner>>>,
}

enum GpuState {
    _Free,
    Acquiring,
    Acquired,
}

struct Gpu {
    prop: comm::CudaDeviceProp,
    runner: Uuid,
    state: Mutex<GpuState>,
}

struct Runner {
    id: Uuid,
    addr: SocketAddr,
    tx: Sender<Message>,
    devices: HashSet<Uuid>,
}

impl Scheduler {
    pub fn new() -> Arc<Scheduler> {
        Arc::new(Scheduler {
            gpus: RwLock::new(HashMap::new()),
            runners: RwLock::new(HashMap::new()),
        })
    }

    pub async fn add_runner(
        &self,
        addr: SocketAddr,
        tx: Sender<Message>,
        msg: comm::AddRunnerRequest,
    ) {
        // Add runner
        let runner = {
            let mut runners = self.runners.write().unwrap();
            if runners.contains_key(&msg.runner_id) {
                error!("Runner {} already exists. Skip.", msg.runner_id);
                return;
            }
            let runner = Arc::new(Runner {
                id: msg.runner_id,
                addr,
                tx,
                devices: msg.devices.iter().map(|prop| prop.uuid).collect(),
            });
            runners.insert(msg.runner_id, runner.clone());
            runner
        };
        let num_gpus = msg.devices.len();
        info!(
            "Add runner {} from {} with {} GPUs",
            runner.id, runner.addr, num_gpus
        );

        // Add GPUs
        {
            let mut gpus = self.gpus.write().unwrap();
            for prop in msg.devices.into_iter() {
                if gpus.contains_key(&prop.uuid) {
                    error!("GPU {} already exists. Skip.", prop.uuid);
                    continue;
                }
                info!(
                    runner = %runner.id,
                    gpu = ?prop,
                    "Add GPU to runner.",
                );
                gpus.insert(
                    prop.uuid,
                    Arc::new(Gpu {
                        prop,
                        runner: msg.runner_id,
                        state: Mutex::new(GpuState::Acquiring),
                    }),
                );
            }
        }

        // Acquire GPUs
        for uuid in runner.devices.iter() {
            runner
                .tx
                .send(Message::Binary(
                    postcard::to_stdvec(
                        &comm::SchedulerToRunnerMessage::AcquireGpuCommand(
                            comm::AcquireGpuCommand { gpu_uuid: *uuid },
                        ),
                    )
                    .unwrap(),
                ))
                .await
                .unwrap();
        }
    }

    pub fn acquire_gpu_resp(&self, msg: comm::AcquireGpuResponse) {
        let gpus = self.gpus.read().unwrap();
        let gpu = gpus.get(&msg.gpu_uuid).unwrap();
        let mut state = gpu.state.lock().unwrap();
        match *state {
            GpuState::Acquiring => {
                *state = GpuState::Acquired;
                info!(
                    "Acquired GPU {} on Runner {}. Name: {}, sm_{}{}, Memory: {:.3} GiB",
                    msg.gpu_uuid,
                    gpu.runner,
                    gpu.prop.name,
                    gpu.prop.sm_major,
                    gpu.prop.sm_minor,
                    gpu.prop.total_memory as f32 / 2f32.powi(30)
                );
            }
            _ => {
                error!("GPU {} is not in Acquiring state. Skip.", msg.gpu_uuid);
            }
        }
    }
}
