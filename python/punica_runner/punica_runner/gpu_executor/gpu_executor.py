import dataclasses
import enum
import gc
import uuid
from typing import TypedDict

import numpy as np
from punica.models.llama import LlamaForCausalLM
from punica.utils import BatchedKvCache
from punica.utils import BatchLenInfo
from punica.utils import KvCache
from punica.utils import KvPool
import torch
import transformers


# Sync with `GenerationConfig` in rust/src/comm.rs.
class GenerationConfig(TypedDict):
  min_tokens: int
  max_tokens: int
  max_new_tokens: int
  stop_token_id: int

  temperature: float
  repetition_penalty: float
  top_p: float


# Sync with `FinishReason` in rust/src/comm.rs.
class FinishReason(enum.Enum):
  NotFinished = 0
  Stop = 1
  Length = 2


# Sync with `GenerationContextChunk` in rust/src/runner/executor.rs.
class TextGenerationChunkResponse(TypedDict):
  request_ids: list[bytes]
  indicies: list[int]
  token_ids: list[int]
  finish_reasons: list[int]
  num_free_kv_blocks: int


class GenerationContext:

  def __init__(
      self,
      input_ids: list[int],
      gencfg: GenerationConfig,
  ):
    self.gencfg = gencfg
    self.output_ids = [int(x) for x in input_ids]
    self.prompt_len = len(self.output_ids)

    self.logits_processor = transformers.LogitsProcessorList()
    if gencfg["temperature"] > 0 and gencfg["temperature"] != 1.0:
      self.logits_processor.append(
          transformers.TemperatureLogitsWarper(gencfg["temperature"]))
    if gencfg["repetition_penalty"] > 1.0:
      self.logits_processor.append(
          transformers.RepetitionPenaltyLogitsProcessor(
              gencfg["repetition_penalty"]))
    if 0 < gencfg["top_p"] < 1.0:
      self.logits_processor.append(
          transformers.TopPLogitsWarper(gencfg["top_p"]))

  def get_next_token_id(self, logits: torch.Tensor) -> int:
    if self.logits_processor:
      if self.gencfg["repetition_penalty"] > 1.0:
        t = torch.as_tensor([self.output_ids], device=logits.device)
      else:
        t = None
      last_token_logits = self.logits_processor(t, logits.unsqueeze(0))[0]
    else:
      last_token_logits = logits

    if self.gencfg["temperature"] <= 0 or self.gencfg["top_p"] <= 0:
      _, indices = torch.topk(last_token_logits, 2)
    else:
      probs = torch.softmax(last_token_logits, dim=-1)
      indices = torch.multinomial(probs, num_samples=2)
    token = int(indices.tolist()[0])
    return token

  def append_token(self, token_id: int):
    self.output_ids.append(token_id)

  def is_finish(self) -> FinishReason:
    outlen = len(self.output_ids)
    if outlen < self.gencfg["min_tokens"]:
      return FinishReason.NotFinished
    if outlen >= self.gencfg["max_tokens"]:
      return FinishReason.Length
    if outlen - self.prompt_len >= self.gencfg["max_new_tokens"]:
      return FinishReason.Length
    if self.output_ids[-1] == self.gencfg["stop_token_id"]:
      return FinishReason.Stop
    return FinishReason.NotFinished


@dataclasses.dataclass
class RequestContext:
  reqid: uuid.UUID
  textgen: GenerationContext
  kvcache: KvCache


class GpuExecutor:

  def __init__(
      self,
      model_path: str,
      dtype_str: str,
      block_len: int,
      kvpool_capacity: int,
  ):
    self.dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype_str]
    self.device = torch.device("cuda:0")
    self.model = LlamaForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        torch_dtype=self.dtype,
    ).to(self.device)

    self.kvpool = KvPool(
        num_layers=self.model.config.num_hidden_layers,
        num_heads=self.model.config.num_attention_heads,
        head_dim=self.model.config.hidden_size //
        self.model.config.num_attention_heads,
        capacity=kvpool_capacity,
        block_len=block_len,
        dtype=self.dtype,
        device=self.device,
    )
    self.reqctx: dict[uuid.UUID, RequestContext] = {}
    self._cnt_step = 0

  def add_request(
      self,
      reqid: uuid.UUID,
      input_ids: list[int],
      gencfg: GenerationConfig,
  ):
    textgen = GenerationContext(input_ids, gencfg)
    kvcache = KvCache(self.kvpool, len(input_ids))
    self.reqctx[reqid] = RequestContext(reqid, textgen, kvcache)

  def _del_request(self, reqid: uuid.UUID):
    self.reqctx[reqid].kvcache.release()
    del self.reqctx[reqid]

  def cancel_request(self, reqid: uuid.UUID):
    self._del_request(reqid)

  def step(self) -> TextGenerationChunkResponse:
    self._cnt_step += 1
    if self._cnt_step % 16 == 0:
      gc.collect()
      torch.cuda.empty_cache()

    prefill_input_ids, prefill_lens, prefill_kv, prefill_reqids = [], [], [], []
    decode_input_ids, decode_kv, decode_reqids = [], [], []
    for reqctx in self.reqctx.values():
      if len(reqctx.textgen.output_ids) == reqctx.textgen.prompt_len:
        prefill_input_ids.extend(reqctx.textgen.output_ids)
        prefill_lens.append(len(reqctx.textgen.output_ids))
        prefill_kv.append(reqctx.kvcache)
        prefill_reqids.append(reqctx.reqid)
      else:
        decode_input_ids.append(reqctx.textgen.output_ids[-1])
        decode_kv.append(reqctx.kvcache)
        decode_reqids.append(reqctx.reqid)
        reqctx.kvcache.acquire_one()

    input_ids = torch.tensor(
        prefill_input_ids + decode_input_ids,
        dtype=torch.long,
        device=self.device)
    blen = BatchLenInfo(prefill_lens, len(decode_input_ids), self.device)
    prefill_kv = BatchedKvCache(prefill_kv) if prefill_kv else None
    decode_kv = BatchedKvCache(decode_kv) if decode_kv else None
    logits, _ = self.model(input_ids, blen, prefill_kv, decode_kv)
    if prefill_kv:
      if decode_kv:
        logits = torch.cat([logits[blen.indptr[1:] - 1], logits[blen.doff:]])
      else:
        logits = logits[blen.indptr[1:] - 1]

    request_ids = prefill_reqids + decode_reqids
    indicies, token_ids, finish_reasons = [], [], []
    for i, reqid in enumerate(request_ids):
      reqctx = self.reqctx[reqid]
      next_token_id = reqctx.textgen.get_next_token_id(logits[i])
      indicies.append(len(reqctx.textgen.output_ids))
      reqctx.textgen.append_token(next_token_id)
      finish = reqctx.textgen.is_finish()
      token_ids.append(next_token_id)
      finish_reasons.append(finish.value)
      if finish != FinishReason.NotFinished:
        self._del_request(reqid)
    return {
        "request_ids": [x.bytes for x in request_ids],
        "indicies": indicies,
        "token_ids": token_ids,
        "finish_reasons": finish_reasons,
        "num_free_kv_blocks": self.kvpool.num_free_blocks,
    }


class FakeGpuExecutor:

  def __init__(self):
    self._reqctx: dict[uuid.UUID, dict] = {}

  def add_request(
      self,
      reqid: uuid.UUID,
      input_ids: list[int],
      gencfg: GenerationConfig,
  ):
    rng = np.random.Generator(np.random.PCG64(seed=sum(input_ids)))
    self._reqctx[reqid] = {
        "gencfg": gencfg,
        "rng": rng,
    }

  def _del_request(self, reqid: uuid.UUID):
    del self._reqctx[reqid]

  def cancel_request(self, reqid: uuid.UUID):
    self._del_request(reqid)

  def step(self) -> TextGenerationChunkResponse:
    request_ids = []
    token_ids = []
    finish_reasons = []
    for reqid, reqctx in self._reqctx.items():
      request_ids.append(reqid.bytes)
      if reqctx["rng"].random() < 0.1:
        next_token_id = reqctx["gencfg"]["stop_token_id"]
        finish = FinishReason.Stop
      else:
        next_token_id = reqctx["rng"].integers(1000, 50000)
        finish = FinishReason.NotFinished
      token_ids.append(int(next_token_id))
      finish_reasons.append(finish.value)
      if finish != FinishReason.NotFinished:
        self._del_request(reqid)
    return {
        "request_ids": request_ids,
        "token_ids": token_ids,
        "finish_reasons": finish_reasons,
        "num_free_kv_blocks": 1000,
    }
