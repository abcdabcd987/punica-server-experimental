mod server;

use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;
use std::sync::{Arc, Mutex, RwLock};

use axum::extract::ws::Message;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::comm;

#[derive(Debug, clap::Args)]
pub struct SchedulerArgs {
    #[arg(long, default_value = "[::1]:23081")]
    pub ws_bind: SocketAddr,
}

pub async fn scheduler_main(args: SchedulerArgs) -> anyhow::Result<()> {
    let ct = CancellationToken::new();
    let http_handle = axum_server::Handle::new();
    let scheduler = Scheduler::new();
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

pub struct Scheduler {
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
    tx: mpsc::Sender<Message>,
    devices: HashSet<Uuid>,
}

impl Runner {
    async fn send_message(&self, msg: &comm::SchedulerToRunnerMessage) {
        self.tx
            .send(Message::Binary(postcard::to_stdvec(msg).unwrap()))
            .await
            .unwrap();
    }
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
        tx: mpsc::Sender<Message>,
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
                .send_message(
                    &comm::SchedulerToRunnerMessage::AcquireGpuCommand(
                        comm::AcquireGpuCommand { gpu_uuid: *uuid },
                    ),
                )
                .await;
        }
    }

    pub fn handle_acquire_gpu_resp(&self, msg: comm::AcquireGpuResponse) {
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
