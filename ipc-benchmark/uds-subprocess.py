import collections
import socket
import struct
import time

import numpy as np


def main():
  sock = socket.socket(socket.AF_UNIX)
  sock.connect(b"\0uds-ipc-benchmark")

  latency_ns = collections.deque(maxlen=1000)
  basetime_ns = 1693420991 * 1000 * 1000 * 1000
  resp = bytearray([0] * 1000)
  sendlen = struct.pack("<I", len(resp))

  last_print_ns = 0
  while True:
    recvlen = struct.unpack("<I", sock.recv(4))[0]
    recv = sock.recv(recvlen)
    recv_time_ns = time.time_ns() - basetime_ns
    server_send_time_ns = struct.unpack("<Q", recv[:8])[0]

    latency_ns.append(recv_time_ns - server_send_time_ns)
    if recv_time_ns - last_print_ns > 50e6:
      l = np.array(latency_ns, dtype=np.float32) / 1e3
      print(
          f"\x1b[2K\rRust Server -> Python Client latency: {l.mean():.3f} us +/- {l.std():.3f} us",
          end="",
          flush=True)
      last_print_ns = recv_time_ns

    send_time_ns = time.time_ns() - basetime_ns
    resp[0:8] = struct.pack("<Q", recv_time_ns)
    resp[8:16] = struct.pack("<Q", send_time_ns)
    sock.sendall(sendlen)
    sock.sendall(resp)


if __name__ == "__main__":
  main()
