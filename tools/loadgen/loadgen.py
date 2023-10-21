import argparse
import dataclasses
from datetime import datetime
import gzip
import json
import pathlib
import subprocess

import numpy as np
import pytz
import scipy.stats
from tqdm import tqdm


def parse_dist(s: str) -> scipy.stats.rv_continuous:
  """
  Supported distribution formats:
    uniform:LOW:HIGH
    lognorm:SHAPE:LOC:SCALE
  """

  split = s.split(":")
  v = [float(x) for x in split[1:]]
  if split[0] == "uniform":
    return scipy.stats.uniform(v[0], v[1] - v[0])
  if split[0] == "lognorm":
    return scipy.stats.lognorm(v[0], v[1], v[2])
  raise ValueError(f"unknown distribution: {s}")


def parse_gap(s: str, rate: float) -> scipy.stats.rv_continuous:
  """
  Supported gap formats:
    exp
  """
  if s == "exp":
    return scipy.stats.expon(scale=1 / rate)


def get_lora_lens(bs: int, popularity: str) -> list[int]:
  if popularity == "identical":
    return [bs]
  if popularity == "distinct":
    return [1] * bs
  if popularity == "uniform":
    n = int(np.ceil(np.sqrt(bs)))
    lens = np.array([bs // n] * n)
    while True:
      diff = bs - lens.sum()
      if diff == 0:
        break
      lens[:abs(diff)] += np.sign(diff)
    return lens.tolist()
  if popularity.startswith("zipf:"):
    alpha = float(popularity.split(":")[1])
    assert alpha > 1
    lens = []
    a = 1
    while sum(lens) + int(np.floor(a)) < bs:
      lens.append(int(np.floor(a)))
      a *= alpha
    lens.append(bs - sum(lens))
    return sorted(lens, reverse=True)
  raise KeyError(popularity)


@dataclasses.dataclass(frozen=True)
class RequestSpec:
  gap: float
  prompt_len: int
  output_len: int
  lora_idx: int

  def to_line(self) -> str:
    return f"{self.gap:.9f} {self.prompt_len} {self.output_len} {self.lora_idx}"


@dataclasses.dataclass
class TraceSpec:
  duration: float
  rps: float
  gap: str
  prompt: str
  output: str
  popularity: str
  seed: int

  def generate(self) -> list[RequestSpec]:
    prompt_dist = parse_dist(self.prompt)
    output_dist = parse_dist(self.output)
    gap_dist = parse_gap(self.gap, self.rps)
    req_rng = np.random.Generator(np.random.PCG64(self.seed))
    gap_rng = np.random.Generator(np.random.PCG64(self.seed + 1))
    pop_rng = np.random.Generator(np.random.PCG64(self.seed + 2))
    t = 0.0
    requests = []
    while t < self.duration:
      gap = gap_dist.rvs(random_state=gap_rng)
      prompt_len = max(int(prompt_dist.rvs(random_state=req_rng)), 2)
      output_len = max(int(output_dist.rvs(random_state=req_rng)), 2)
      requests.append((gap, prompt_len, output_len))
      t += gap
    lora_lens = get_lora_lens(len(requests), self.popularity)
    pop_rng.shuffle(lora_lens)
    requests = [
        RequestSpec(gap, prompt_len, output_len, lora_idx)
        for (gap, prompt_len, output_len), lora_idx in zip(requests, lora_lens)
    ]
    return requests


def main():
  this_file = pathlib.Path(__file__)
  project_root = this_file.resolve().parents[2]
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--bin",
      type=pathlib.Path,
      default=project_root / "target/release/punica")
  parser.add_argument("--host", type=str, default="localhost")
  parser.add_argument("--port", type=int, default=23081)
  parser.add_argument("--warmup", type=int, default=10)
  parser.add_argument("--bench", type=int, default=60)
  parser.add_argument("--rps", type=float, default=1)
  parser.add_argument("--gap", type=str, default="exp")
  parser.add_argument("--prompt", type=str, default="lognorm:0.8:-1:18")
  parser.add_argument("--output", type=str, default="uniform:1:2048")
  parser.add_argument("--popularity", default="identical")
  parser.add_argument("--seed", type=int, default=0xabcdabcd987)
  parser.add_argument("--output-bins", type=int, default=100)
  parser.add_argument("--print-trace", action="store_true")
  args = parser.parse_args()

  total_duration = args.warmup + args.bench
  trace_spec = TraceSpec(
      duration=total_duration,
      rps=args.rps,
      gap=args.gap,
      prompt=args.prompt,
      output=args.output,
      popularity=args.popularity,
      seed=args.seed,
  )
  trace = trace_spec.generate()
  if args.print_trace:
    for req in trace:
      print(req.to_line())
    exit(0)

  cmd = [
      args.bin,
      "loadgen",
      f"--scheduler-url=ws://{args.host}:{args.port}/rpc",
  ]
  popen = subprocess.Popen(
      cmd,
      bufsize=1,
      text=True,
      stdin=subprocess.PIPE,
      stdout=subprocess.PIPE,
      restore_signals=False,
  )
  for req in trace:
    popen.stdin.write(req.to_line())
    popen.stdin.write("\n")
  popen.stdin.close()

  assert popen.stdout.readline().strip() == "start"

  nbins = args.output_bins
  bin_duration = args.bench / nbins
  bins = [dict(tokens=0, gpus=dict()) for _ in range(nbins)]

  pbar = tqdm(unit="tok")
  sigint = False
  done_prefill = set()
  while popen.poll() is None:
    try:
      line = popen.stdout.readline().strip()
    except KeyboardInterrupt:
      sigint = True
      break
    if not line:
      continue
    split = line.split(" ")
    elapsed = float(split[0])
    reqidx = int(split[1])
    gpu_uuid = split[2]
    new_tokens = 1
    if reqidx not in done_prefill:
      new_tokens += trace[reqidx].prompt_len
      done_prefill.add(reqidx)
    pbar.update(new_tokens)
    if elapsed > total_duration:
      break
    bin_idx = int(np.floor((elapsed - args.warmup) / bin_duration))
    if 0 <= bin_idx < nbins:
      bin = bins[bin_idx]
      bin["tokens"] += new_tokens
      if gpu_uuid not in bin["gpus"]:
        bin["gpus"][gpu_uuid] = set()
      bin["gpus"][gpu_uuid].add(reqidx)

  pbar.close()
  popen.terminate()
  ec = popen.wait()
  if ec == 0 and sigint:
    ec = 1

  gpus = set(gpu for bin in bins for gpu in bin["gpus"])
  gpu_batch_size_bins = {
      gpu: [len(bin["gpus"].get(gpu, set())) for bin in bins] for gpu in gpus
  }
  throughput_bins = [bin["tokens"] / bin_duration for bin in bins]
  result = {
      "setup": {
          "warmup": args.warmup,
          "bench": args.bench,
          "rps": args.rps,
          "gap": args.gap,
          "prompt": args.prompt,
          "output": args.output,
          "popularity": args.popularity,
          "seed": args.seed,
      },
      "gpu_batch_size_bins": gpu_batch_size_bins,
      "throughput_bins": throughput_bins,
  }
  now = datetime.now(pytz.timezone("US/Pacific"))
  out_filename = f"{now:%Y%m%d-%H%M%S}-{this_file.stem}.jsonl.gz"
  out_path = project_root / "data" / out_filename
  out_path.parent.mkdir(parents=True, exist_ok=True)
  with gzip.open(out_path, "wt") as f:
    json.dump(result, f)

  if ec != 0:
    exit(ec)


if __name__ == "__main__":
  main()
