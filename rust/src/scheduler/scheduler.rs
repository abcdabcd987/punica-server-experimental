use std::collections::{BTreeSet, HashMap, HashSet};

use uuid::Uuid;

use super::traits::{RequestStub, RunnerStub};
use crate::comm;
use crate::model_config::LlamaModelConfig;

pub struct Scheduler<R: RunnerStub, Q: RequestStub> {
    model_config: LlamaModelConfig,
    runners: HashMap<Uuid, R>,
    gpus: HashMap<Uuid, GpuContext>,
    requests: HashMap<Uuid, RequestContext<Q>>,

    /// (batch_size, gpu_uuid)
    /// This scheduler always try to use the GPU with the largest batch_size.
    gpus_accepting_new_requests: BTreeSet<(u32, Uuid)>,
}

struct GpuContext {
    gpu_uuid: Uuid,
    runner_id: Uuid,
    devprop: comm::CudaDeviceProp,
    max_batch_size: u32,
    state: GpuState,
}

enum GpuState {
    Invalid,
    Initing,
    Running(RunningGpu),
}

struct RunningGpu {
    requests: HashSet<Uuid>,
}

struct RequestContext<Q: RequestStub> {
    request: Q,
    gpu_uuid: Uuid,
    len: u32,
}

impl<R: RunnerStub, Q: RequestStub> Scheduler<R, Q> {
    pub fn new(model_config: LlamaModelConfig) -> Self {
        Self {
            model_config,
            runners: HashMap::new(),
            gpus: HashMap::new(),
            requests: HashMap::new(),
            gpus_accepting_new_requests: BTreeSet::new(),
        }
    }

    pub fn add_runner(&mut self, runner: R) {
        if self.runners.contains_key(&runner.id()) {
            error!(runner_id=%runner.id(), "Runner already exists. Skip.");
            return;
        }
        let num_gpus = runner.device_props().len();
        info!(runner_id=%runner.id(), addr=%runner.addr(), num_gpus=%num_gpus, "Add runner.");

        for prop in runner.device_props().iter() {
            if self.gpus.contains_key(&prop.uuid) {
                error!(gpu_uuid=%prop.uuid, "GPU already exists. Skip.");
                continue;
            }

            let sizeof = 2;
            let block_len = 16;
            let mut mem = (prop.total_memory as f32 * 0.8) as i64;
            mem -= (self.model_config.total_params() * sizeof) as i64;
            let block_size =
                self.model_config.token_kvcache_size() * block_len * sizeof;
            let kvpool_capacity = mem / block_size as i64;
            let maxlen = 2048;
            let max_batch_size = kvpool_capacity / (maxlen / block_len) as i64;
            if max_batch_size <= 0 {
                error!(gpu_uuid=%prop.uuid, gpu_name=%prop.name, gpu_total_memory=%prop.total_memory, "Not enough memory. Skip.");
                self.gpus.insert(
                    prop.uuid,
                    GpuContext {
                        gpu_uuid: prop.uuid,
                        runner_id: runner.id(),
                        devprop: prop.clone(),
                        max_batch_size: 0,
                        state: GpuState::Invalid,
                    },
                );
                continue;
            }
            let max_batch_size = u32::max(max_batch_size as u32, 32);
            self.gpus.insert(
                prop.uuid,
                GpuContext {
                    gpu_uuid: prop.uuid,
                    runner_id: runner.id(),
                    devprop: prop.clone(),
                    max_batch_size,
                    state: GpuState::Initing,
                },
            );
            runner.init_gpu(comm::AcquireGpuCommand {
                gpu_uuid: prop.uuid,
                dtype: "float16".to_string(),
                block_len: block_len as u32,
                kvpool_capacity: kvpool_capacity as u32,
            });
        }

        self.runners.insert(runner.id(), runner);
    }

    pub fn notify_gpu_initialized(&mut self, msg: &comm::AcquireGpuResponse) {
        let gpu = self.gpus.get_mut(&msg.gpu_uuid).unwrap();
        match &mut gpu.state {
            GpuState::Initing => (),
            _ => panic!("Invalid state."),
        }

        gpu.state = GpuState::Running(RunningGpu { requests: HashSet::new() });
        self.gpus_accepting_new_requests.insert((0, gpu.gpu_uuid));
        info!(runner_id=%gpu.runner_id, gpu_uuid=%gpu.gpu_uuid, gpu_name=%gpu.devprop.name, "GPU initialized.");
    }

    pub fn del_runner(&mut self, runner_id: Uuid) {
        let runner = match self.runners.remove(&runner_id) {
            Some(v) => v,
            None => {
                error!(runner_id=%runner_id, "Runner not found. Skip.");
                return;
            }
        };
        for prop in runner.device_props().iter() {
            let gpu = match self.gpus.remove(&prop.uuid) {
                Some(v) => v,
                None => {
                    error!(gpu_uuid=%prop.uuid, "GPU not found. Skip.");
                    continue;
                }
            };
            if let GpuState::Running(state) = gpu.state {
                let found = self
                    .gpus_accepting_new_requests
                    .remove(&(state.requests.len() as u32, gpu.gpu_uuid));
                if !found {
                    panic!("Corrupted gpus_accepting_new_requests");
                }
                for request_id in state.requests {
                    let reqctx = self.requests.remove(&request_id).unwrap();
                    reqctx.request.add_chunk(0, comm::FinishReason::Error);
                }
            }
            info!(runner_id=%runner_id, gpu_uuid=%gpu.gpu_uuid, gpu_name=%gpu.devprop.name, "GPU removed.");
        }
        info!(runner_id=%runner_id, addr=%runner.addr(), "Runner removed.");
    }

    pub fn add_textgen(&mut self, request: Q) -> bool {
        let (_, gpu_uuid) = match self.gpus_accepting_new_requests.pop_last() {
            Some(v) => v,
            None => {
                error!(reqid=%request.id(), "Unable to schedule textgen. No GPU available.");
                return false;
            }
        };
        let gpu = self.gpus.get_mut(&gpu_uuid).unwrap();
        let gpu_state = gpu.state.unwrap_running_mut();
        let runner = self.runners.get(&gpu.runner_id).unwrap();

        let request_id = request.id();
        gpu_state.requests.insert(request_id);
        if (gpu_state.requests.len() as u32) < gpu.max_batch_size {
            self.gpus_accepting_new_requests
                .insert((gpu_state.requests.len() as u32, gpu.gpu_uuid));
        }
        runner.run_textgen(comm::RunTextGenCommand {
            gpu_uuid,
            req: comm::TextGenRequest {
                request_id,
                input_ids: request.input_ids().to_vec(),
                gencfg: request.generation_config().clone(),
            },
        });
        let len = request.input_ids().len() as u32;
        self.requests
            .insert(request_id, RequestContext { request, gpu_uuid, len });
        true
    }

    pub fn notify_textgen_chunk(&mut self, msg: &comm::BatchedTextGenChunk) {
        for chunk in &msg.chunks {
            let reqctx = match self.requests.get_mut(&chunk.request_id) {
                Some(v) => v,
                None => {
                    error!(reqid=%chunk.request_id, "Request not found. Skip.");
                    continue;
                }
            };
            let gpu = self.gpus.get_mut(&reqctx.gpu_uuid).unwrap();
            let gpu_state = gpu.state.unwrap_running_mut();

            reqctx.request.add_chunk(chunk.token_id, chunk.finish_reason);
            reqctx.len += 1;

            if chunk.finish_reason != comm::FinishReason::NotFinished {
                let found = self
                    .gpus_accepting_new_requests
                    .remove(&(gpu_state.requests.len() as u32, gpu.gpu_uuid));
                if !found {
                    panic!("Corrupted gpus_accepting_new_requests");
                }
                gpu_state.requests.remove(&chunk.request_id);
                self.gpus_accepting_new_requests
                    .insert((gpu_state.requests.len() as u32, gpu.gpu_uuid));
            }
        }
    }

    pub fn cancel_textgen(&mut self, msg: &comm::CancelTextGen) {
        let reqctx = match self.requests.get_mut(&msg.request_id) {
            Some(v) => v,
            None => {
                error!(reqid=%msg.request_id, "Request not found. Skip.");
                return;
            }
        };
        let gpu = self.gpus.get_mut(&reqctx.gpu_uuid).unwrap();
        let gpu_state = gpu.state.unwrap_running_mut();

        let runner = self.runners.get(&gpu.runner_id).unwrap();
        runner.cancel_textgen(comm::CancelTextGenCommand {
            gpu_uuid: gpu.gpu_uuid,
            request_id: msg.request_id,
        });

        let found = self
            .gpus_accepting_new_requests
            .remove(&(gpu_state.requests.len() as u32, gpu.gpu_uuid));
        if !found {
            panic!("Corrupted gpus_accepting_new_requests");
        }
        gpu_state.requests.remove(&msg.request_id);
        self.gpus_accepting_new_requests
            .insert((gpu_state.requests.len() as u32, gpu.gpu_uuid));
        gpu_state.requests.remove(&msg.request_id);
        reqctx.request.add_chunk(0, comm::FinishReason::Stop);
    }
}

impl GpuState {
    fn unwrap_running_mut(&mut self) -> &mut RunningGpu {
        match self {
            GpuState::Running(v) => v,
            _ => panic!("Invalid state."),
        }
    }
}
