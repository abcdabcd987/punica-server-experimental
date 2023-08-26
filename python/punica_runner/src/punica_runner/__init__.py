import torch
import pynvml
import multiprocessing


def _run_in_subprocess_worker(q, target, args, kwargs):
  q.put(target(*args, **kwargs))


def run_in_subprocess(target, args=(), kwargs={}):
  q = multiprocessing.Queue()
  p = multiprocessing.Process(
      target=_run_in_subprocess_worker, args=(q, target, args, kwargs))
  p.start()
  p.join()
  return q.get()


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


def hello():
  print("Hello, world! from python")
  print("pytorch version", torch.__version__)
  print("cuda available", torch.cuda.is_available())


def dev_main():
  gpu_info_list = run_in_subprocess(get_all_gpu_info)
  for i, gpu_info in enumerate(gpu_info_list):
    mem_size, uuid, gpu_name, sm_major, sm_minor = gpu_info
    print(f"GPU {i}: "
          f"UUID: {uuid} "
          f"Memory size: {mem_size} bytes "
          f"GPU name: {gpu_name} "
          f"sm_major: {sm_major} "
          f"sm_minor: {sm_minor} ")
