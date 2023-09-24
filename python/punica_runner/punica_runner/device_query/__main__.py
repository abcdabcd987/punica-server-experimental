import os
import sys
from uuid import UUID

import msgpack
import pynvml
import torch.cuda


def get_cuda_uuids() -> list[UUID]:
  from ctypes import byref
  from ctypes import c_int
  from ctypes import CDLL
  from ctypes import create_string_buffer
  cudart = CDLL("libcudart.so")
  cnt = c_int()
  assert cudart.cudaGetDeviceCount(byref(cnt)) == 0
  prop = create_string_buffer(1024)
  ret = []
  for i in range(cnt.value):
    assert cudart.cudaGetDeviceProperties(byref(prop), i) == 0
    ret.append(UUID(bytes=prop.raw[256:272]))
  return ret


def get_all_gpu_info():
  uuids = get_cuda_uuids()

  gpu_info_list = []
  for i, uuid in enumerate(uuids):
    prop = torch.cuda.get_device_properties(i)
    gpu_info_list.append(
        dict(
            uuid=uuid.bytes,
            name=prop.name,
            total_memory=prop.total_memory,
            sm_major=prop.major,
            sm_minor=prop.minor,
        ))
  return gpu_info_list


def main():
  stdout = os.fdopen(sys.stdout.fileno(), "wb")
  gpu_info_list = get_all_gpu_info()
  stdout.write(msgpack.packb(gpu_info_list))


if __name__ == "__main__":
  main()
