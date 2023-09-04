use serde::{Deserialize, Serialize};
use uuid::Uuid;

//====== Common ======

#[derive(Serialize, Deserialize, Debug)]
pub struct TextGenParams {
    pub n: u32,
    pub min_tokens: u32,
    pub max_tokens: u32,
    pub seed: u64,
    pub temperature: f32,
    pub top_p: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub ignore_eos: bool,
    pub stop: Vec<String>,
}

//====== TextGen ======

#[derive(Serialize, Deserialize, Debug)]
pub struct TextGenRequest {
    pub request_id: Uuid,
    pub prompt: String,
    pub params: TextGenParams,
}

#[derive(Serialize, Deserialize, Debug)]
pub enum FinishReason {
    Stop,
    Length,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ChoiceChunk {
    pub index: u32,
    pub text: String,
    pub finish_reason: FinishReason,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct TextGenChunk {
    pub request_id: Uuid,
    pub choices: Vec<ChoiceChunk>,
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

    TextGenChunk(TextGenChunk),
    UpdateGpuStatsRequest(UpdateGpuStatsRequest),
}

#[derive(Serialize, Deserialize, Debug)]
pub enum SchedulerToRunnerMessage {
    RunnerExitCommand(RunnerExitCommand),
    AcquireGpuCommand(AcquireGpuCommand),
    ReleaseGpuCommand(ReleaseGpuCommand),

    TextGenRequest(TextGenRequest),
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

//====== HelloScheduler ======

#[derive(Serialize, Deserialize, Debug)]
pub enum NodeType {
    Runner,
    ApiServer,
    Scheduler,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct HelloScheduler {
    pub node_type: NodeType,
}
