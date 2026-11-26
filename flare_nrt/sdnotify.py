"""systemd notifications (1.3.1), standard library only: ``READY=1`` once
the model is loaded, ``WATCHDOG=1`` whenever the loop makes progress
(after each download pass, each slot, each replay build). With
``Type=notify`` and ``WatchdogSec=600`` (deploy/flare-nrt-loop.service),
systemd restarts a loop that stops making progress for 10 min, whatever
it is stuck in. Without ``NOTIFY_SOCKET`` (tests, a shell) it does nothing."""

from __future__ import annotations

import os
import socket


def notify(state: str) -> bool:
    """Sends ``state`` (e.g. ``"WATCHDOG=1"``) to systemd; True if sent."""
    addr = os.environ.get("NOTIFY_SOCKET")
    if not addr:
        return False
    if addr[0] == "@":                        # abstract namespace
        addr = "\0" + addr[1:]
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM | socket.SOCK_CLOEXEC) as s:
            s.connect(addr)
            s.sendall(state.encode())
        return True
    except OSError:
        return False
