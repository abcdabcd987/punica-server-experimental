#![allow(dead_code)]

#[derive(Debug, Clone)]
pub struct LlamaModelConfig {
    pub num_layers: u32,
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub hidden_dim: u32,
    pub intermediate_size: u32,
    pub vocab_size: u32,
}

impl LlamaModelConfig {
    pub const fn total_params(&self) -> u64 {
        let q =
            self.hidden_dim as u64 * (self.num_heads * self.head_dim) as u64;
        let k =
            self.hidden_dim as u64 * (self.num_kv_heads * self.head_dim) as u64;
        let v = k;
        let o = q;
        let gate = self.hidden_dim as u64 * self.intermediate_size as u64;
        let up = gate;
        let down = gate;

        let layer = q + k + v + o + gate + up + down;
        let embed = self.vocab_size as u64 * self.hidden_dim as u64;
        let lm = embed;

        embed + layer * self.num_layers as u64 + lm
    }

    pub const fn token_kvcache_size(&self) -> u64 {
        let k = (self.num_kv_heads * self.head_dim) as u64;
        k * self.num_layers as u64 * 2
    }

    pub const fn lora_params(&self, r: u32) -> u64 {
        let q = lora_params(self.hidden_dim, self.num_heads * self.head_dim, r);
        let k =
            lora_params(self.hidden_dim, self.num_kv_heads * self.head_dim, r);
        let v = k;
        let o = q;
        let gate = lora_params(self.hidden_dim, self.intermediate_size, r);
        let up = gate;
        let down = gate;
        let layer = q + k + v + o + gate + up + down;

        layer * self.num_layers as u64
    }
}

const fn lora_params(d_in: u32, d_out: u32, r: u32) -> u64 {
    (d_in + d_out) as u64 * r as u64
}

pub const LLAMA_7B: LlamaModelConfig = LlamaModelConfig {
    num_layers: 32,
    num_heads: 32,
    num_kv_heads: 32,
    head_dim: 128,
    hidden_dim: 4096,
    intermediate_size: 11008,
    vocab_size: 32000,
};

pub const LLAMA_13B: LlamaModelConfig = LlamaModelConfig {
    num_layers: 40,
    num_heads: 40,
    num_kv_heads: 40,
    head_dim: 128,
    hidden_dim: 5120,
    intermediate_size: 13824,
    vocab_size: 32000,
};

pub const LLAMA_70B: LlamaModelConfig = LlamaModelConfig {
    num_layers: 80,
    num_heads: 64,
    num_kv_heads: 8,
    head_dim: 128,
    hidden_dim: 8192,
    intermediate_size: 28672,
    vocab_size: 32000,
};
