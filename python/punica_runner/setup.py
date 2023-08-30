import pathlib

import setuptools
import torch.utils.cpp_extension as torch_cpp_ext

root = pathlib.Path(__name__).parent
generated_kernels = [str(p) for p in root.glob("csrc/gen/*.cu")]


def remove_unwanted_pytorch_nvcc_flags():
  REMOVE_NVCC_FLAGS = [
      '-D__CUDA_NO_HALF_OPERATORS__',
      '-D__CUDA_NO_HALF_CONVERSIONS__',
      '-D__CUDA_NO_BFLOAT16_CONVERSIONS__',
      '-D__CUDA_NO_HALF2_OPERATORS__',
  ]
  for flag in REMOVE_NVCC_FLAGS:
    try:
      torch_cpp_ext.COMMON_NVCC_FLAGS.remove(flag)
    except ValueError:
      pass


remove_unwanted_pytorch_nvcc_flags()
ext_modules = []
ext_modules.append(
    torch_cpp_ext.CUDAExtension(
        "punica_runner._kernels",
        ["csrc/punica_ops.cc"] + generated_kernels,
    ))

setuptools.setup(
    name="punica_runner",
    ext_modules=ext_modules,
    cmdclass={"build_ext": torch_cpp_ext.BuildExtension},
)
