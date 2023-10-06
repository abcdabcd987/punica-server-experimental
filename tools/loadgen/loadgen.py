import argparse
import dataclasses
import pathlib
import subprocess

import numpy as np
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
    return scipy.stats.expon(1 / rate)


@dataclasses.dataclass(frozen=True)
class RequestSpec:
  gap: float
  prompt_len: int
  output_len: int


@dataclasses.dataclass
class TraceSpec:
  duration: float
  rps: float
  gap: str
  prompt: str
  output: str
  seed: int

  def generate(self) -> list[RequestSpec]:
    prompt_dist = parse_dist(self.prompt)
    output_dist = parse_dist(self.output)
    gap_dist = parse_gap(self.gap, self.rps)
    req_rng = np.random.Generator(np.random.PCG64(self.seed))
    gap_rng = np.random.Generator(np.random.PCG64(self.seed + 1))
    t = 0.0
    requests = []
    while t < self.duration:
      gap = gap_dist.rvs(random_state=gap_rng)
      prompt_len = int(prompt_dist.rvs(random_state=req_rng))
      output_len = int(output_dist.rvs(random_state=req_rng))
      requests.append(RequestSpec(gap, prompt_len, output_len))
      t += gap
    return requests


def main():
  project_root = pathlib.Path(__file__).resolve().parents[2]
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
  parser.add_argument("--seed", type=int, default=0xabcdabcd987)
  parser.add_argument("--print-trace", action="store_true")
  args = parser.parse_args()

  total_duration = args.warmup + args.bench
  trace_spec = TraceSpec(
      duration=total_duration,
      rps=args.rps,
      gap=args.gap,
      prompt=args.prompt,
      output=args.output,
      seed=args.seed,
  )
  trace = trace_spec.generate()
  if args.print_trace:
    for req in trace:
      print(f"{req.gap:.9f} {req.prompt_len} {req.output_len}")
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
    popen.stdin.write(f"{req.gap:.9f} {req.prompt_len} {req.output_len}\n")
  popen.stdin.close()

  assert popen.stdout.readline().strip() == "start"

  pbar = tqdm(unit="tok")
  sigint = False
  done_prefill = set()
  while popen.poll() is None:
    try:
      line = popen.stdout.readline().strip()
    except KeyboardInterrupt:
      sigint = True
      break
    split = line.split(" ")
    elapsed = float(split[0])
    reqidx = int(split[1])
    if reqidx not in done_prefill:
      pbar.update(trace[reqidx].prompt_len)
      done_prefill.add(reqidx)
    pbar.update()
    if elapsed > total_duration:
      break
  pbar.close()
  popen.terminate()
  popen.wait()
  if sigint:
    exit(1)


if __name__ == "__main__":
  main()
