import logging

try:
  logging.disable()
  from deepspeed.utils.logging import logger
  logger.handlers.clear()
except ImportError:
  pass

logging.disable(logging.NOTSET)

import os
import struct
import sys
import traceback
import uuid

import msgpack
from punica_runner.gpu_executor.gpu_executor import GpuExecutor

stdin = os.fdopen(sys.stdin.fileno(), "rb")
stdout = os.fdopen(sys.stdout.fileno(), "wb")
exe: GpuExecutor | None = None


def read_msg():
  read_len = struct.unpack("<I", stdin.read(4))[0]
  content = stdin.read(read_len)
  return msgpack.loads(content)


def write_msg(obj):
  content = msgpack.dumps(obj)
  write_len = struct.pack("<I", len(content))
  stdout.write(write_len)
  stdout.write(content)
  stdout.flush()


def handle_init(msg):
  global exe
  exe = GpuExecutor(
      model_path=msg["model_path"],
      dtype_str=msg["dtype_str"],
      block_len=msg["block_len"],
      kvpool_capacity=msg["kvpool_capacity"],
      lora_cache_size=msg["lora_cache_size"],
      lora_rank=msg["lora_rank"],
  )
  return 0


def handle_add_request(msg):
  exe.add_request(
      reqid=uuid.UUID(bytes=msg["reqid"]),
      lora_id=uuid.UUID(bytes=msg["lora_id"]),
      input_ids=msg["input_ids"],
      gencfg=msg["gencfg"],
  )
  return 0


def handle_cancel_request(msg):
  exe.cancel_request(uuid.UUID(bytes=msg["reqid"]))
  return 0


def handle_step(_msg):
  return exe.step()


def do_main():
  HANDLERS = {
      "Init": handle_init,
      "AddRequest": handle_add_request,
      "CancelRequest": handle_cancel_request,
      "Step": handle_step,
  }
  while True:
    msg = read_msg()
    cmd, content = msg["t"], msg["c"]
    if cmd == "Shutdown":
      break
    handler = HANDLERS[cmd]
    ret = handler(content)
    write_msg({"Ok": ret})


def main_wrapper():
  try:
    do_main()
  except Exception as e:
    write_msg({"Err": "".join(traceback.format_exception(e))})
    raise e
  except KeyboardInterrupt:
    return


if __name__ == "__main__":
  main_wrapper()
