use serde::{Deserialize, Serialize};
use serde_repr::{Deserialize_repr, Serialize_repr};
use uuid::Uuid;

//====== Common ======

#[derive(Serialize, Deserialize, Debug)]
pub enum NodeType {
    Runner,
    ApiServer,
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
    pub input_ids: Vec<u32>,
    pub gencfg: GenerationConfig,
}

#[derive(Serialize_repr, Deserialize_repr, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum FinishReason {
    NotFinished = 0,
    Stop = 1,
    Length = 2,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct TextGenChunk {
    pub request_id: Uuid,
    pub token_id: u32,
    pub finish_reason: FinishReason,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CancelTextGen {
    pub request_id: Uuid,
}

//====== Runner ======

#[derive(Serialize, Deserialize, Debug)]
pub enum RunnerToSchedulerMessage {
    AddRunnerRequest(AddRunnerRequest),
    DelRunnerRequest(DelRunnerRequest),
    AcquireGpuResponse(AcquireGpuResponse),
    ReleaseGpuResponse(ReleaseGpuResponse),
    RunnerMigrateToNewSchedulerCommand(RunnerMigrateToNewSchedulerCommand),
    RunnerMigratedToNewScheduler(RunnerMigratedToNewScheduler),

    BatchedTextGenChunk(BatchedTextGenChunk),
    UpdateGpuStatsRequest(UpdateGpuStatsRequest),
}

#[derive(Serialize, Deserialize, Debug)]
pub enum SchedulerToRunnerMessage {
    RunnerExitCommand(RunnerExitCommand),
    AcquireGpuCommand(AcquireGpuCommand),
    ReleaseGpuCommand(ReleaseGpuCommand),

    RunTextGenCommand(RunTextGenCommand),
    CancelTextGen(CancelTextGen),
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
}

#[derive(Serialize, Deserialize, Debug)]
pub struct DelRunnerRequest {}

#[derive(Serialize, Deserialize, Debug)]
pub struct RunnerExitCommand {}

#[derive(Serialize, Deserialize, Debug)]
pub struct AcquireGpuCommand {
    pub gpu_uuid: Uuid,
    pub dtype: String,
    pub block_len: u32,
    pub kvpool_capacity: u32,
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
    pub request_id: Uuid,
    pub input_ids: Vec<u32>,
    pub gencfg: GenerationConfig,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct BatchedTextGenChunk {
    pub chunks: Vec<TextGenChunk>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateGpuStatsRequest {
    pub gpu_uuid: Uuid,
    pub batch_size: u32,
    pub memory_used: u64,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RunnerMigrateToNewSchedulerCommand {
    pub new_host: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RunnerMigratedToNewScheduler {}

//====== ApiServer ======

#[derive(Serialize, Deserialize, Debug)]
pub enum ApiServerToSchedulerMessage {
    AddApiServerRequest(AddApiServerRequest),
    DelApiServerRequest(DelApiServerRequest),
    ApiServerMigratedToNewScheduler(ApiServerMigratedToNewScheduler),

    TextGenRequest(TextGenRequest),
    CancelTextGen(CancelTextGen),
}

#[derive(Serialize, Deserialize, Debug)]
pub enum SchedulerToApiServerMessage {
    ApiServerExitCommand(ApiServerExitCommand),
    ApiServerMigrateToNewSchedulerCommand(
        ApiServerMigrateToNewSchedulerCommand,
    ),

    TextGenChunk(TextGenChunk),
}

#[derive(Serialize, Deserialize, Debug)]
pub struct AddApiServerRequest {
    pub api_server_id: Uuid,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct DelApiServerRequest {}

#[derive(Serialize, Deserialize, Debug)]
pub struct ApiServerExitCommand {}

#[derive(Serialize, Deserialize, Debug)]
pub struct ApiServerMigrateToNewSchedulerCommand {
    pub new_host: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ApiServerMigratedToNewScheduler {}

//====== Scheduler ======

#[derive(Serialize, Deserialize, Debug)]
pub enum SchedulerToSchedulerMessage {
    SchedulerTakeOverRequest(SchedulerTakeOverRequest),
    SchedulerTakeOverResponse(SchedulerTakeOverResponse),
}

#[derive(Serialize, Deserialize, Debug)]
pub struct SchedulerTakeOverRequest {}

#[derive(Serialize, Deserialize, Debug)]
pub struct SchedulerTakeOverResponse {}
