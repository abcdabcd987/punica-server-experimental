import struct
import time
import os
import sys


def main():
  stdin = os.fdopen(sys.stdin.fileno(), "rb")
  stdout = os.fdopen(sys.stdout.fileno(), "wb")
  basetime_ns = 1693420991 * 1000 * 1000 * 1000
  resp = bytearray([0] * 1000)
  sendlen = struct.pack("<I", len(resp))

  while True:
    recvlen = struct.unpack("<I", stdin.read(4))[0]
    recv = stdin.read(recvlen)
    recv_time_ns = time.time_ns() - basetime_ns
    server_send_time_ns = struct.unpack("<Q", recv[:8])[0]

    send_time_ns = time.time_ns() - basetime_ns
    resp[0:8] = struct.pack("<Q", recv_time_ns)
    resp[8:16] = struct.pack("<Q", send_time_ns)
    stdout.write(sendlen)
    stdout.write(resp)
    stdout.flush()


if __name__ == "__main__":
  main()
