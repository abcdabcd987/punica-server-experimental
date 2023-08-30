import dataclasses
import functools
import multiprocessing

import pynvml
import torch
import transformers

from .models.llama import LlamaConfig
from .models.llama import LlamaForCausalLM
from .utils import CatTensor
from .utils import KvPool


def _run_in_subprocess_worker(q, f, args, kwargs):
  q.put(f(*args, **kwargs))


def run_in_subprocess(f):

  @functools.wraps(f)
  def wrapper(*args, **kwargs):
    q = multiprocessing.Queue()
    p = multiprocessing.Process(
        target=_run_in_subprocess_worker, args=(q, f, args, kwargs))
    p.start()
    p.join()
    return q.get()

  return wrapper


@run_in_subprocess
def get_all_gpu_info():
  pynvml.nvmlInit()
  cnt = pynvml.nvmlDeviceGetCount()
  gpu_info_list = []
  for i in range(cnt):
    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    uuid = pynvml.nvmlDeviceGetUUID(handle).removeprefix("GPU-")
    name = pynvml.nvmlDeviceGetName(handle)
    sm_major, sm_minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
    gpu_info_list.append((uuid, name, meminfo.total, sm_major, sm_minor))
  pynvml.nvmlShutdown()
  return gpu_info_list


def prepare_logits_processor(
    temperature: float,
    repetition_penalty: float,
    top_p: float,
    top_k: int,
) -> transformers.LogitsProcessorList:
  processor_list = transformers.LogitsProcessorList()
  if temperature >= 1e-5 and temperature != 1.0:
    processor_list.append(transformers.TemperatureLogitsWarper(temperature))
  if repetition_penalty > 1.0:
    processor_list.append(
        transformers.RepetitionPenaltyLogitsProcessor(repetition_penalty))
  if 1e-8 <= top_p < 1.0:
    processor_list.append(transformers.TopPLogitsWarper(top_p))
  if top_k > 0:
    processor_list.append(transformers.TopKLogitsWarper(top_k))
  return processor_list


@dataclasses.dataclass
class GenerationContext:
  temperature: float
  repetition_penalty: float
  top_p: float
  top_k: int
  max_new_tokens: int
  stop_token_ids: list[int]
  logits_processor: transformers.LogitsProcessorList

  kv_idx: int
  prompt_len: int
  pastlen: int
  next_token_id: int
  output_ids: list[int]


class PunicaRunner:

  def __init__(self, model_path: str, dtype: str, max_batch_size: int):
    self.model_config: LlamaConfig = LlamaConfig.from_pretrained(model_path)
    self.device = torch.device("cuda:0")
    self.dtype = getattr(torch, dtype)
    self.max_seqlen = 2048

    self.model = LlamaForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        torch_dtype=self.dtype,
    ).to(self.device)
    self.kvpool = KvPool(
        num_layers=self.model_config.num_hidden_layers,
        num_heads=self.model_config.num_attention_heads,
        head_dim=self.model_config.hidden_size //
        self.model_config.num_attention_heads,
        capacity=max_batch_size,
        block_len=self.max_seqlen,
        dtype=self.dtype,
        device=self.device,
    )
    self.genctx: list[GenerationContext | None] = [None] * self.kvpool.capacity
    self.available_genctx: set[int] = set(range(self.kvpool.capacity))

  def _get_next_token_id(self, ctx: GenerationContext,
                         logits: torch.Tensor) -> int:
    if ctx.logits_processor:
      if ctx.repetition_penalty > 1.0:
        t = torch.as_tensor([ctx.output_ids], device=logits.device)
      else:
        t = None
      last_token_logits = ctx.logits_processor(t, logits[-1, :].unsqueeze(0))[0]
    else:
      last_token_logits = logits[-1, :]

    if ctx.temperature < 1e-5 or ctx.top_p < 1e-8:
      _, indices = torch.topk(last_token_logits, 2)
    else:
      probs = torch.softmax(last_token_logits, dim=-1)
      indices = torch.multinomial(probs, num_samples=2)
    token = int(indices.tolist()[0])
    return token

  def _is_stop(self, ctx: GenerationContext) -> int:
    if len(ctx.output_ids) - ctx.prompt_len >= ctx.max_new_tokens:
      return 1
    if ctx.next_token_id in ctx.stop_token_ids:
      return 2
    return 0

  def _free_genctx(self, genidx: int):
    ctx = self.genctx[genidx]
    self.genctx[genidx] = None
    self.available_genctx.add(genidx)
    self.kvpool.free_block(ctx.kv_idx)

  @torch.inference_mode()
  def prefill(
      self,
      input_ids: list[int],
      temperature=1.0,
      repetition_penalty=1.0,
      top_p=1.0,
      top_k=-1,
      max_new_tokens=500,
      stop_token_ids: list[int] | None = None,
  ) -> tuple[int, int, int]:
    genidx = self.available_genctx.pop()
    kv_idx = self.kvpool.alloc_block()
    stop_token_ids = stop_token_ids or []
    logits_processor = prepare_logits_processor(temperature, repetition_penalty,
                                                top_p, top_k)
    prompt_len = len(input_ids)
    ctx = GenerationContext(
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        top_p=top_p,
        top_k=top_k,
        max_new_tokens=min(max_new_tokens, self.max_seqlen - prompt_len - 1),
        stop_token_ids=stop_token_ids,
        logits_processor=logits_processor,
        kv_idx=kv_idx,
        prompt_len=prompt_len,
        pastlen=0,
        next_token_id=0,
        output_ids=list(input_ids),
    )
    self.genctx[genidx] = ctx

    input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device)
    past_lens = torch.tensor([0], dtype=torch.long, device=self.device)
    kvidx = torch.tensor([kv_idx], dtype=torch.long, device=self.device)
    logits, _h = self.model(self.kvpool, CatTensor(input_ids, [prompt_len]),
                            past_lens, kvidx)
    next_token_id = self._get_next_token_id(ctx, logits.cat)

    ctx.pastlen = prompt_len
    ctx.next_token_id = next_token_id
    ctx.output_ids.append(next_token_id)
    stop = self._is_stop(ctx)
    if stop:
      self._free_genctx(genidx)
    return genidx, next_token_id, stop

  @torch.inference_mode()
  def batch_decode(self, genidxs: list[int]) -> list[tuple[int, int]]:
    input_ids = CatTensor(
        torch.tensor([self.genctx[idx].next_token_id for idx in genidxs],
                     dtype=torch.long,
                     device=self.device), [1] * len(genidxs))
    past_lens = torch.tensor([self.genctx[idx].pastlen for idx in genidxs],
                             dtype=torch.long,
                             device=self.device)
    kvidx = torch.tensor([self.genctx[idx].kv_idx for idx in genidxs],
                         dtype=torch.long,
                         device=self.device)
    logits, _h = self.model(self.kvpool, input_ids, past_lens, kvidx)

    outputs = []
    # TODO: Do we need to improve logit processing? Benchmark.
    for i, genidx in enumerate(genidxs):
      ctx = self.genctx[genidx]
      next_token_id = self._get_next_token_id(ctx, logits.cat[i:i + 1])
      ctx.pastlen += 1
      ctx.next_token_id = next_token_id
      ctx.output_ids.append(next_token_id)
      stop = self._is_stop(ctx)
      outputs.append((next_token_id, stop))
      if stop:
        self._free_genctx(genidx)
    return outputs


def textgen_demo():
  from rich.console import Console
  from rich.layout import Layout
  from rich.live import Live
  from rich.panel import Panel
  from rich.text import Text

  model_path = "/dataset/hf/Llama-2-7b-chat-hf"
  questions = [
      "Give me a 3 day travel plan for Seattle.",
      "Tell me something about University of Washington.",
      "How to dial in an espresso shot?",
  ]
  prompt_tpl = '[INST] <<SYS>> You are a helpful, respectful and honest assistant. <</SYS>>\n{prompt} [/INST]\n'

  tokenizer = transformers.AutoTokenizer.from_pretrained(
      model_path, use_fast=True)
  runner = PunicaRunner(
      model_path=model_path,
      dtype="float16",
      max_batch_size=len(questions),
  )
  console = Console()
  layout = Layout()
  layout.split_row(*[Layout(name=str(i)) for i in range(len(questions))])

  with Live(layout, console=console, auto_refresh=False) as live:
    # Prompt
    output_ids = []
    for i, question in enumerate(questions):
      out = tokenizer(prompt_tpl.format(prompt=question)).input_ids
      text = tokenizer.decode(
          out,
          skip_special_tokens=True,
          spaces_between_special_tokens=False,
          clean_up_tokenization_spaces=True,
      )
      layout[str(i)].update(Panel(Text(text, overflow="ellipsis")))
      output_ids.append(out)
    live.refresh()

    # Prefill
    genidxs = []
    workset = set()
    for i in range(len(questions)):
      out = output_ids[i]
      genidx, next_id, stop = runner.prefill(
          out,
          temperature=0.7,
          repetition_penalty=1.1,
          top_p=0.9,
          stop_token_ids=[tokenizer.eos_token_id],
          max_new_tokens=600,
      )
      genidxs.append(genidx)
      out.append(next_id)
      text = tokenizer.decode(
          out,
          skip_special_tokens=True,
          spaces_between_special_tokens=False,
          clean_up_tokenization_spaces=True,
      )
      layout[str(i)].update(Panel(Text(text, overflow="ellipsis")))
      if stop == 0:
        workset.add(i)
      live.refresh()

    # Decode
    while workset:
      ret = runner.batch_decode([genidxs[i] for i in workset])
      for i, (next_id, stop) in zip(workset.copy(), ret):
        output_ids[i].append(next_id)
        text = tokenizer.decode(
            output_ids[i],
            skip_special_tokens=True,
            spaces_between_special_tokens=False,
            clean_up_tokenization_spaces=True,
        )
        if stop != 0:
          workset.remove(i)
          text += "\n\nstop reason: "
          if stop == 1:
            text += "max_new_tokens"
          elif stop == 2:
            text += "stop_token_ids"
        layout[str(i)].update(Panel(Text(text, overflow="ellipsis")))
      live.refresh()


def dev_main():
  textgen_demo()
