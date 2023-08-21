import torch


def hello():
  print("Hello, world! from python")
  print("pytorch version", torch.__version__)
  print("cuda available", torch.cuda.is_available())
