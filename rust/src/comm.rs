use serde::{Deserialize, Serialize};
use serde_repr::{Deserialize_repr, Serialize_repr};
use uuid::Uuid;

//====== Common ======

#[derive(Serialize, Deserialize, Debug)]
pub enum NodeType {
    Runner,
    Frontend,
    Scheduler,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GenerationConfig {
    pub min_tokens: u32,
    pub max_tokens: u32,
    pub max_new_tokens: u32,
    pub stop_token_id: u32,

    pub temperature: f32,
    pub repetition_penalty: f32,
    pub top_p: f32,
}

//====== TextGen ======

#[derive(Serialize, Deserialize, Debug)]
pub struct TextGenRequest {
    pub request_id: Uuid,
    pub lora_id: Uuid,
    pub input_ids: Vec<u32>,
    pub gencfg: GenerationConfig,
}

#[derive(
    Serialize_repr, Deserialize_repr, Debug, PartialEq, Eq, Clone, Copy,
)]
#[repr(u8)]
pub enum FinishReason {
    NotFinished = 0,
    Stop = 1,
    Length = 2,
    Error = 3,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TextGenChunk {
    pub request_id: Uuid,
    pub index: u32,
    pub token_id: u32,
    pub finish_reason: FinishReason,
}

//====== Runner ======

#[derive(Serialize, Deserialize, Debug)]
pub enum RunnerToSchedulerMessage {
    AddRunnerRequest(AddRunnerRequest),
    AcquireGpuResponse(AcquireGpuResponse),
    ReleaseGpuResponse(ReleaseGpuResponse),

    BatchedTextGenChunk(BatchedTextGenChunk),
}

#[derive(Serialize, Deserialize, Debug)]
pub enum SchedulerToRunnerMessage {
    AcquireGpuCommand(AcquireGpuCommand),
    ReleaseGpuCommand(ReleaseGpuCommand),

    RunTextGenCommand(RunTextGenCommand),
    CancelTextGen(CancelTextGenCommand),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CudaDeviceProp {
    pub uuid: Uuid,
    pub name: String,
    pub total_memory: u64,
    pub sm_major: i8,
    pub sm_minor: i8,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct AddRunnerRequest {
    pub runner_id: Uuid,
    pub devices: Vec<CudaDeviceProp>,
    pub limit_gpumem: Option<u64>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct AcquireGpuCommand {
    pub gpu_uuid: Uuid,
    pub dtype: String,
    pub block_len: u32,
    pub kvpool_capacity: u32,
    pub lora_cache_size: u32,
    pub lora_rank: u32,
}
#[derive(Serialize, Deserialize, Debug)]
pub struct AcquireGpuResponse {
    pub gpu_uuid: Uuid,
}
#[derive(Serialize, Deserialize, Debug)]
pub struct ReleaseGpuCommand {
    pub gpu_uuid: Uuid,
}
#[derive(Serialize, Deserialize, Debug)]
pub struct ReleaseGpuResponse {
    pub gpu_uuid: Uuid,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RunTextGenCommand {
    pub gpu_uuid: Uuid,
    pub req: TextGenRequest,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CancelTextGenCommand {
    pub gpu_uuid: Uuid,
    pub request_id: Uuid,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct BatchedTextGenChunk {
    pub chunks: Vec<TextGenChunk>,
    pub gpu_uuid: Uuid,
    pub num_free_kv_blocks: u32,
}

//====== Frontend ======

#[derive(Serialize, Deserialize, Debug)]
pub enum FrontendToSchedulerMessage {
    TextGenRequest(TextGenRequest),
    CancelTextGen(CancelTextGen),
}

#[derive(Serialize, Deserialize, Debug)]
pub enum SchedulerToFrontendMessage {
    TextGenChunk(TextGenChunk),
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CancelTextGen {
    pub request_id: Uuid,
}
