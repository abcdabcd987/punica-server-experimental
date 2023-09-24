use std::collections::{BinaryHeap, HashMap, HashSet};
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};

use uuid::Uuid;

use crate::comm;
use crate::model_config::LlamaModelConfig;

pub trait RunnerStub {
    // Properties
    fn id(&self) -> Uuid;
    fn device_props(&self) -> &[comm::CudaDeviceProp];
    fn addr(&self) -> SocketAddr;

    // Commands
    fn init_gpu(&self, msg: comm::AcquireGpuCommand);
    fn run_textgen(&self, msg: comm::RunTextGenCommand);
    fn cancel_textgen(&self, msg: comm::CancelTextGen);
}

pub trait RequestStub {
    // Properties
    fn id(&self) -> Uuid;
    fn input_ids(&self) -> &[u32];
    fn generation_config(&self) -> &comm::GenerationConfig;

    // Commands
    fn add_chunk(&self, token_id: u32, finish: comm::FinishReason);
}

pub struct Scheduler<R: RunnerStub, Q: RequestStub> {
    model_config: LlamaModelConfig,
    runners: HashMap<Uuid, R>,
    gpus: HashMap<Uuid, GpuContext>,
    requests: HashMap<Uuid, RequestContext<Q>>,
    gpus_accepting_new_requests: BinaryHeap<(u32, Uuid)>,
}

struct GpuContext {
    gpu_uuid: Uuid,
    runner_id: Uuid,
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
            gpus_accepting_new_requests: BinaryHeap::new(),
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
            let mut mem = (prop.total_memory as f32 * 0.9) as i64;
            mem -= (self.model_config.total_params() * sizeof) as i64;
            let block_size =
                self.model_config.token_kvcache_size() * block_len * sizeof;
            let kvpool_capacity = mem / block_size as i64;
            let max_batch_size = kvpool_capacity / 2048;
            if max_batch_size <= 0 {
                error!(gpu_uuid=%prop.uuid, gpu_name=%prop.name, gpu_total_memory=%prop.total_memory, "Not enough memory. Skip.");
                self.gpus.insert(
                    prop.uuid,
                    GpuContext {
                        gpu_uuid: prop.uuid,
                        runner_id: runner.id(),
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
        self.gpus_accepting_new_requests.push((0, gpu.gpu_uuid));
    }

    pub fn add_textgen(&mut self, request: Q) -> bool {
        let (_, gpu_uuid) = match self.gpus_accepting_new_requests.pop() {
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
        self.gpus_accepting_new_requests
            .push((gpu_state.requests.len() as u32, gpu.gpu_uuid));
        runner.run_textgen(comm::RunTextGenCommand {
            gpu_uuid,
            request_id,
            input_ids: request.input_ids().to_vec(),
            gencfg: request.generation_config().clone(),
        });
        let len = request.input_ids().len() as u32;
        self.requests
            .insert(request_id, RequestContext { request, gpu_uuid, len });
        true
    }

    pub fn notify_textgen_chunk(&mut self, msg: &comm::BatchedTextGenChunk) {
        todo!();
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
