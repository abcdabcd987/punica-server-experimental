import dataclasses
import enum
import uuid
from typing import TypedDict

from punica.models.llama import LlamaForCausalLM
from punica.utils import BatchedKvCache
from punica.utils import CatTensor
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
  token_ids: list[int]
  finish_reasons: list[int]


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
      last_token_logits = self.logits_processor(t, logits[-1].unsqueeze(0))[0]
    else:
      last_token_logits = logits[-1, :]

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

  def batch_prefill(self, reqs: list[uuid.UUID]) -> TextGenerationChunkResponse:
    # NOTE: Waiting for FlashInfer to add batch_prefill.
    #       Use for loop for now.
    token_ids = []
    finish_reasons = []
    for reqid in reqs:
      reqctx = self.reqctx[reqid]
      input_ids = torch.tensor(
          reqctx.textgen.output_ids, dtype=torch.long, device=self.device)
      logits, _ = self.model(
          input_ids=CatTensor(input_ids, [reqctx.textgen.prompt_len]),
          kv=BatchedKvCache([reqctx.kvcache]),
          is_decode=False,
      )
      next_token_id = reqctx.textgen.get_next_token_id(logits.cat)
      reqctx.textgen.append_token(next_token_id)
      finish = reqctx.textgen.is_finish()
      token_ids.append(next_token_id)
      finish_reasons.append(finish.value)
      if finish != FinishReason.NotFinished:
        self._del_request(reqid)
    return {
        "token_ids": token_ids,
        "finish_reasons": finish_reasons,
    }

  def batch_decode(self, reqs: list[uuid.UUID]) -> TextGenerationChunkResponse:
    input_ids = []
    kv = []
    for reqid in reqs:
      reqctx = self.reqctx[reqid]
      reqctx.kvcache.acquire_one()
      input_ids.append(reqctx.textgen.output_ids[-1])
      kv.append(reqctx.kvcache)
    input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device)
    logits, _ = self.model(
        input_ids=CatTensor(input_ids, [1] * len(reqs)),
        kv=BatchedKvCache(kv),
        is_decode=True,
    )

    token_ids = []
    finish_reasons = []
    for i in range(len(reqs)):
      reqctx = self.reqctx[reqs[i]]
      next_token_id = reqctx.textgen.get_next_token_id(logits.cat[i:i + 1])
      reqctx.textgen.append_token(next_token_id)
      finish = reqctx.textgen.is_finish()
      token_ids.append(next_token_id)
      finish_reasons.append(finish.value)
      if finish != FinishReason.NotFinished:
        self._del_request(reqs[i])
    return {
        "token_ids": token_ids,
        "finish_reasons": finish_reasons,
    }
