import os
import sys
from uuid import UUID

import msgpack
import pynvml


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
    gpu_info_list.append(
        dict(
            uuid=UUID(uuid).bytes,
            name=name,
            total_memory=meminfo.total,
            sm_major=sm_major,
            sm_minor=sm_minor,
        ))
  pynvml.nvmlShutdown()
  return gpu_info_list


def main():
  stdout = os.fdopen(sys.stdout.fileno(), "wb")
  gpu_info_list = get_all_gpu_info()
  stdout.write(msgpack.packb(gpu_info_list))


if __name__ == "__main__":
  main()
