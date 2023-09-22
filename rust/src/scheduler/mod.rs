mod server;

use std::collections::HashSet;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};

use axum::extract::ws::Message;
use dashmap::DashMap;
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
    gpus: DashMap<Uuid, Arc<Gpu>>,
    runners: DashMap<Uuid, Arc<Runner>>,
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
        Arc::new(Scheduler { gpus: DashMap::new(), runners: DashMap::new() })
    }

    pub async fn add_runner(
        &self,
        addr: SocketAddr,
        tx: mpsc::Sender<Message>,
        msg: comm::AddRunnerRequest,
    ) {
        // Add runner
        if self.runners.contains_key(&msg.runner_id) {
            error!("Runner {} already exists. Skip.", msg.runner_id);
            return;
        }
        let runner = Arc::new(Runner {
            id: msg.runner_id,
            addr,
            tx,
            devices: msg.devices.iter().map(|prop| prop.uuid).collect(),
        });
        self.runners.insert(msg.runner_id, runner.clone());
        let num_gpus = msg.devices.len();
        info!(
            "Add runner {} from {} with {} GPUs",
            runner.id, runner.addr, num_gpus
        );

        // Add GPUs
        for prop in msg.devices.into_iter() {
            if self.gpus.contains_key(&prop.uuid) {
                error!("GPU {} already exists. Skip.", prop.uuid);
                continue;
            }
            info!(
                runner = %runner.id,
                gpu = ?prop,
                "Add GPU to runner.",
            );
            self.gpus.insert(
                prop.uuid,
                Arc::new(Gpu {
                    prop,
                    runner: msg.runner_id,
                    state: Mutex::new(GpuState::Acquiring),
                }),
            );
        }

        // Acquire GPUs
        for uuid in runner.devices.iter() {
            runner
                .send_message(
                    &comm::SchedulerToRunnerMessage::AcquireGpuCommand(
                        comm::AcquireGpuCommand {
                            gpu_uuid: *uuid,
                            dtype: "".to_string(),
                            block_len: 0,
                            kvpool_capacity: 0,
                        },
                    ),
                )
                .await;
        }
    }

    pub fn handle_acquire_gpu_resp(&self, msg: comm::AcquireGpuResponse) {
        let gpu = self.gpus.get(&msg.gpu_uuid).unwrap();
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
