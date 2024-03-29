use serde::{Deserialize, Serialize};

#[derive(Serialize, Debug)]
pub struct Error {
    pub message: String,
}

#[derive(Deserialize, Debug)]
pub struct TextGenRequest {
    pub prompt: String,
    pub template: Option<String>,

    pub min_tokens: Option<u32>,
    pub max_tokens: Option<u32>,
    pub max_new_tokens: Option<u32>,

    pub temperature: Option<f32>,
    pub repetition_penalty: Option<f32>,
    pub top_p: Option<f32>,
}

#[derive(Serialize, Debug, Clone, Copy)]
pub enum FinishReason {
    #[serde(rename = "stop")]
    Stop,
    #[serde(rename = "length")]
    Length,
    #[serde(rename = "error")]
    Error,
}

#[derive(Serialize, Debug)]
pub struct TextGenChunk {
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,
}
