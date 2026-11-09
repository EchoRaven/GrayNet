"""Small CONNECT-only provider proxy; the agent has no direct egress route.

No TLS interception and no general browsing. Destinations must be explicitly
allowed provider hosts on port 443 and resolve only to public addresses.
"""

import ipaddress
import json
import os
import select
import socket
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def allowed_destination(host: str, port: int, allowed_hosts: set[str]) -> bool:
    return port == 443 and host.casefold() in allowed_hosts


class ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args: object) -> None:
        pass

    def do_CONNECT(self) -> None:
        host, separator, port_text = self.path.rpartition(":")
        port = int(port_text) if separator and port_text.isdecimal() else 0
        hosts = set(os.environ.get("GRAYNET_PROVIDER_HOSTS", "").split(","))
        if not allowed_destination(host, port, hosts):
            print(json.dumps({"event": "blocked", "host": host, "port": port}), flush=True)
            self.send_error(403, "Destination is not an allowed provider")
            return
        try:
            addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
            if not addresses or any(
                not ipaddress.ip_address(entry[4][0]).is_global for entry in addresses
            ):
                self.send_error(403, "Provider resolved to a non-public address")
                return
            # Connect to the validated address, not a second DNS resolution.
            family, sock_type, proto, _, address = next(
                (entry for entry in addresses if entry[0] == socket.AF_INET), addresses[0]
            )
            with socket.socket(family, sock_type, proto) as upstream:
                upstream.settimeout(15)
                upstream.connect(address)
                self.send_response(200, "Connection established")
                self.end_headers()
                self.wfile.flush()
                sockets = (self.connection, upstream)
                deadline = time.monotonic() + 600
                while time.monotonic() < deadline:
                    ready, _, _ = select.select(sockets, [], [], 15)
                    for source in ready:
                        data = source.recv(65536)
                        if not data:
                            self.close_connection = True
                            return
                        target = upstream if source is self.connection else self.connection
                        target.sendall(data)
        except (OSError, ValueError):
            self.close_connection = True

    def do_GET(self) -> None:
        self.send_error(403, "Only provider HTTPS tunnels are allowed")

    do_POST = do_GET


if __name__ == "__main__":
    server = ThreadingHTTPServer(("0.0.0.0", 8080), ProxyHandler)
    server.daemon_threads = True
    server.serve_forever()
