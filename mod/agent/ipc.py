# Low-level JSON-lines IPC client for the DD2 plugin.
# Provides socket connection, send, receive, and timeout behavior for live control.
# Shared by mod-side baseline and live environment wrappers.

from __future__ import annotations

import json
import socket
import time
from typing import Any


class NdjsonClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 8765, timeout: float = 8.0) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self._sock: socket.socket | None = None
        self._buffer = b""

    def connect(self) -> None:
        self.close()
        self._sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        self._sock.settimeout(self.timeout)
        self._buffer = b""

    def ensure_connected(self) -> None:
        if self._sock is None:
            self.connect()

    def send(self, message: dict[str, Any]) -> None:
        self.ensure_connected()
        data = (json.dumps(message, ensure_ascii=True) + "\n").encode("utf-8")
        try:
            assert self._sock is not None
            self._sock.sendall(data)
        except (BrokenPipeError, ConnectionResetError, OSError):
            self.connect()
            assert self._sock is not None
            self._sock.sendall(data)

    def recv_lines(self, timeout: float, wait_cycles: int = 1) -> list[dict[str, Any]]:
        self.ensure_connected()
        assert self._sock is not None
        per_recv_timeout = max(0.05, float(timeout))
        self._sock.settimeout(per_recv_timeout)
        messages: list[dict[str, Any]] = []
        idle_cycles = 0
        deadline = time.monotonic() + (per_recv_timeout * max(1, int(wait_cycles)) + 0.5)

        while idle_cycles < wait_cycles and time.monotonic() < deadline:
            try:
                chunk = self._sock.recv(8192)
            except (TimeoutError, socket.timeout):
                idle_cycles += 1
                continue
            except (ConnectionResetError, OSError):
                self.connect()
                break

            if not chunk:
                self.close()
                break

            idle_cycles = 0
            self._buffer += chunk

            while b"\n" in self._buffer:
                raw, self._buffer = self._buffer.split(b"\n", 1)
                if not raw.strip():
                    continue
                try:
                    messages.append(json.loads(raw.decode("utf-8")))
                except json.JSONDecodeError:
                    # Ignore malformed line and continue consuming stream.
                    continue

            if messages:
                break

        return messages

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
        self._sock = None
        self._buffer = b""

