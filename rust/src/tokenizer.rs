use std::path::Path;

use anyhow::{anyhow, Context};

pub struct Tokenizer {
    tokenizer: tokenizers::Tokenizer,
    eos_id: u32,
}

impl Tokenizer {
    pub fn new(model_path: &Path) -> anyhow::Result<Self> {
        let tokenizer =
            tokenizers::Tokenizer::from_file(model_path.join("tokenizer.json"))
                .map_err(|e| anyhow!("{}", e))
                .with_context(|| {
                    format!(
                        "Failed to load tokenizer from {}",
                        model_path.display()
                    )
                })?;

        let file =
            std::fs::read_to_string(model_path.join("tokenizer_config.json"))
                .with_context(|| {
                format!(
                    "Failed to read tokenizer_config.json from {}",
                    model_path.display()
                )
            })?;
        let json: serde_json::Value = serde_json::from_str(&file)?;
        let eos_id = json["eos_token"]["content"]
            .as_str()
            .ok_or_else(|| {
                anyhow!("Cannot read eos_token from tokenizer_config.json")
            })
            .and_then(|s| {
                tokenizer.encode(s, false).map_err(|e| anyhow!("{}", e))
            })?
            .get_ids()[0];

        Ok(Self { tokenizer, eos_id })
    }

    pub fn eos_id(&self) -> u32 {
        self.eos_id
    }

    pub fn encode(&self, text: &str) -> anyhow::Result<Vec<u32>> {
        self.tokenizer
            .encode(text, false)
            .map_err(|e| anyhow!("{}", e))
            .map(|encodings| encodings.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32]) -> anyhow::Result<String> {
        self.tokenizer.decode(ids, true).map_err(|e| anyhow!("{}", e))
    }
}
