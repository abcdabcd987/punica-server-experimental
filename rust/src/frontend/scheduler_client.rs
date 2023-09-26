use tokio::sync::mpsc;
use uuid::Uuid;

use crate::comm;

#[derive(Clone)]
pub struct SchedulerClient {}

impl SchedulerClient {
    pub fn add_textgen(
        &self,
        request_id: Uuid,
        input_ids: Vec<u32>,
        generation_config: comm::GenerationConfig,
        chunk_tx: mpsc::UnboundedSender<comm::TextGenChunk>,
    ) {
        todo!();
    }

    pub fn cancel_textgen(&self, request_id: Uuid) {
        todo!();
    }
}
