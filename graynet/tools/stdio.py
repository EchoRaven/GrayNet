"""Public MCP transport only: no evaluator imports or case files in the image."""

import socket
import sys

MAX_MESSAGE = 65536


def main() -> None:
    for line in sys.stdin.buffer:
        if len(line) > MAX_MESSAGE:
            raise ValueError("MCP message exceeds transport limit")
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(30)
            connection.connect(sys.argv[1])
            connection.sendall(line)
            response = connection.makefile("rb").readline(MAX_MESSAGE + 1)
        if response and response.strip():
            sys.stdout.buffer.write(response)
            sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
