use std::collections::{HashMap, HashSet};

use uuid::Uuid;

use super::traits::{RequestStub, RunnerStub};
use crate::comm;
use crate::model_config::LlamaModelConfig;
use crate::paged_kv_tracker::PagedKvTracker;

pub struct Scheduler<R: RunnerStub, Q: RequestStub> {
    model_config: LlamaModelConfig,
    runners: HashMap<Uuid, R>,
    gpus: HashMap<Uuid, GpuContext>,
    frontends: HashMap<Uuid, FrontendContext>,
    requests: HashMap<Uuid, RequestContext<Q>>,
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
    Initing(InitingGpu),
    Running(RunningGpu),
}

struct InitingGpu {
    kvpool_capacity: u32,
    kv_block_len: u32,
}

struct RunningGpu {
    requests: HashSet<Uuid>,
    kvpool: PagedKvTracker,
}

struct RequestContext<Q: RequestStub> {
    request: Q,
    gpu_uuid: Uuid,
    len: u32,
}

struct FrontendContext {
    request_ids: HashSet<Uuid>,
}

impl<R: RunnerStub, Q: RequestStub> Scheduler<R, Q> {
    pub fn new(model_config: LlamaModelConfig) -> Self {
        Self {
            model_config,
            runners: HashMap::new(),
            gpus: HashMap::new(),
            frontends: HashMap::new(),
            requests: HashMap::new(),
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
            let max_batch_size = u32::min(max_batch_size as u32, 32);
            self.gpus.insert(
                prop.uuid,
                GpuContext {
                    gpu_uuid: prop.uuid,
                    runner_id: runner.id(),
                    devprop: prop.clone(),
                    max_batch_size,
                    state: GpuState::Initing(InitingGpu {
                        kvpool_capacity: kvpool_capacity as u32,
                        kv_block_len: block_len as u32,
                    }),
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
        let init = match &gpu.state {
            GpuState::Initing(v) => v,
            _ => panic!("Invalid state."),
        };

        let kvpool =
            PagedKvTracker::new(init.kvpool_capacity, init.kv_block_len);
        gpu.state =
            GpuState::Running(RunningGpu { requests: HashSet::new(), kvpool });
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
                for request_id in state.requests {
                    let reqctx = self.requests.remove(&request_id).unwrap();
                    if let Some(fctx) =
                        self.frontends.get_mut(&reqctx.request.frontend_id())
                    {
                        fctx.request_ids.remove(&request_id);
                    }
                    reqctx.request.add_chunk(0, comm::FinishReason::Error);
                }
            }
            info!(runner_id=%runner_id, gpu_uuid=%gpu.gpu_uuid, gpu_name=%gpu.devprop.name, "GPU removed.");
        }
        info!(runner_id=%runner_id, addr=%runner.addr(), "Runner removed.");
    }

    fn get_gpu_for_new_request(&mut self, seqlen: u32) -> Option<Uuid> {
        let mut best = None;
        for (_, gpu) in self.gpus.iter_mut() {
            let gpu_state = gpu.state.unwrap_running_mut();
            if gpu_state.requests.len() as u32 >= gpu.max_batch_size {
                continue;
            }
            let free_blocks = gpu_state.kvpool.num_free_blocks();
            let new_blocks = gpu_state.kvpool.calc_init_blocks(seqlen);
            if gpu_state.kvpool.num_free_blocks() < new_blocks {
                continue;
            }

            let candidate = (
                gpu_state.requests.len() as u32,
                -(free_blocks as i32),
                gpu.gpu_uuid,
            );
            best = best.max(Some(candidate));
        }

        best.map(|(_, _, gpu_uuid)| gpu_uuid)
    }

    pub fn add_textgen(&mut self, request: Q) -> bool {
        let prompt_len = request.input_ids().len() as u32;
        let gpu_uuid = match self.get_gpu_for_new_request(prompt_len) {
            Some(v) => v,
            None => {
                warn!(reqid=%request.id(), "Unable to schedule textgen. No GPU available.");
                return false;
            }
        };
        let gpu = self.gpus.get_mut(&gpu_uuid).unwrap();
        let gpu_state = gpu.state.unwrap_running_mut();
        let runner = self.runners.get(&gpu.runner_id).unwrap();

        let request_id = request.id();
        gpu_state.requests.insert(request_id);
        debug!(%request_id, %gpu_uuid, gpu_batch_size=%gpu_state.requests.len(), "Add textgen request.");
        runner.run_textgen(comm::RunTextGenCommand {
            gpu_uuid,
            req: comm::TextGenRequest {
                request_id,
                input_ids: request.input_ids().to_vec(),
                gencfg: request.generation_config().clone(),
            },
        });
        self.frontends
            .entry(request.frontend_id())
            .or_insert_with(|| FrontendContext { request_ids: HashSet::new() })
            .request_ids
            .insert(request_id);
        self.requests.insert(
            request_id,
            RequestContext { request, gpu_uuid, len: prompt_len },
        );
        true
    }

    pub fn notify_textgen_chunk(&mut self, msg: &comm::BatchedTextGenChunk) {
        let gpu = self.gpus.get_mut(&msg.gpu_uuid).unwrap();
        let gpu_state = gpu.state.unwrap_running_mut();

        // TODO: a better way to handle cancelled requests.
        let mut some_missing = false;

        for chunk in &msg.chunks {
            let reqctx = match self.requests.get_mut(&chunk.request_id) {
                Some(v) => v,
                None => {
                    warn!(reqid=%chunk.request_id, "Request not found. Skip.");
                    some_missing = true;
                    continue;
                }
            };

            if reqctx.len == reqctx.request.input_ids().len() as u32 {
                assert!(gpu_state.kvpool.init(chunk.request_id, reqctx.len));
            } else {
                gpu_state.kvpool.append_token(&chunk.request_id);
            }

            reqctx.request.add_chunk(chunk.token_id, chunk.finish_reason);
            reqctx.len += 1;

            if chunk.finish_reason != comm::FinishReason::NotFinished {
                if let Some(fctx) =
                    self.frontends.get_mut(&reqctx.request.frontend_id())
                {
                    fctx.request_ids.remove(&chunk.request_id);
                }

                assert!(gpu_state.requests.remove(&chunk.request_id));
                self.requests.remove(&chunk.request_id);
                debug!(request_id=%chunk.request_id, gpu_uuid=%gpu.gpu_uuid, gpu_batch_size=%gpu_state.requests.len(), "Textgen finished.");

                gpu_state.kvpool.release(chunk.request_id);
            }
        }

        if !some_missing {
            assert_eq!(
                gpu_state.kvpool.num_free_blocks(),
                msg.num_free_kv_blocks
            );
        }
    }

    pub fn cancel_textgen_internal(
        &mut self,
        msg: &comm::CancelTextGen,
        send_result: bool,
    ) {
        let reqctx = match self.requests.remove(&msg.request_id) {
            Some(v) => v,
            None => {
                warn!(reqid=%msg.request_id, "Request not found. Skip.");
                return;
            }
        };
        let gpu = self.gpus.get_mut(&reqctx.gpu_uuid).unwrap();
        let gpu_state = gpu.state.unwrap_running_mut();

        if let Some(fctx) =
            self.frontends.get_mut(&reqctx.request.frontend_id())
        {
            fctx.request_ids.remove(&msg.request_id);
        }

        let runner = self.runners.get(&gpu.runner_id).unwrap();
        runner.cancel_textgen(comm::CancelTextGenCommand {
            gpu_uuid: gpu.gpu_uuid,
            request_id: msg.request_id,
        });

        assert!(gpu_state.requests.remove(&msg.request_id));
        debug!(request_id=%msg.request_id, gpu_uuid=%gpu.gpu_uuid, gpu_batch_size=%gpu_state.requests.len(), "Cancel textgen.");

        gpu_state.kvpool.release(msg.request_id);

        if send_result {
            reqctx.request.add_chunk(0, comm::FinishReason::Stop);
        }
    }

    pub fn cancel_textgen(&mut self, msg: &comm::CancelTextGen) {
        self.cancel_textgen_internal(msg, true);
    }

    pub fn del_frontend(&mut self, frontend_id: Uuid) {
        let fctx = match self.frontends.remove(&frontend_id) {
            Some(v) => v,
            None => {
                error!(frontend_id=%frontend_id, "Frontend not found. Skip.");
                return;
            }
        };
        for request_id in fctx.request_ids {
            info!(%request_id, "Cancel request because of frontend removal.");
            self.cancel_textgen_internal(
                &comm::CancelTextGen { request_id },
                false,
            );
        }
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
