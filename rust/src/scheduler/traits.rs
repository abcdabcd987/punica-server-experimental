use std::net::SocketAddr;

use uuid::Uuid;

use crate::comm;

pub trait RunnerStub {
    // Properties
    fn id(&self) -> Uuid;
    fn device_props(&self) -> &[comm::CudaDeviceProp];
    fn addr(&self) -> SocketAddr;
    fn limit_gpumem(&self) -> Option<u64>;

    // Commands
    fn init_gpu(&self, msg: comm::AcquireGpuCommand);
    fn run_textgen(&self, msg: comm::RunTextGenCommand);
    fn cancel_textgen(&self, msg: comm::CancelTextGenCommand);
}

pub trait RequestStub {
    // Properties
    fn id(&self) -> Uuid;
    fn input_ids(&self) -> &[u32];
    fn generation_config(&self) -> &comm::GenerationConfig;
    fn tokens(&self) -> &[u32];
    fn len(&self) -> u32;
    fn frontend_id(&self) -> Uuid;

    // Commands
    fn add_chunk(&mut self, token_id: u32, finish: comm::FinishReason);
    fn migrate(&mut self);
}
