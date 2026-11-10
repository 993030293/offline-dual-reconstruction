"""Deny non-loopback Python networking in reconstruction child processes."""

import ipaddress
import json
import os
from pathlib import Path
import socket
import threading
import time

_lock = threading.Lock()
_connect = socket.socket.connect
_connect_ex = socket.socket.connect_ex
_sendto = socket.socket.sendto
_lookup = socket.getaddrinfo


def _local(address):
    if not isinstance(address, tuple):
        return True
    host = address[0]
    if host in ("localhost", b"localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _require_local(address):
    if _local(address):
        return
    destination = os.environ.get("RECON_OFFLINE_DENIAL_LOG")
    if destination:
        with _lock:
            with Path(destination).open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"unix": time.time(), "pid": os.getpid(),
                    "status": "denied_external_network", "host": str(address[0])}) + "\n")
    raise PermissionError("Offline reconstruction denies non-loopback network destinations")


def _guarded_connect(instance, address):
    _require_local(address)
    return _connect(instance, address)


def _guarded_connect_ex(instance, address):
    _require_local(address)
    return _connect_ex(instance, address)


def _guarded_sendto(instance, data, *arguments):
    _require_local(arguments[-1])
    return _sendto(instance, data, *arguments)


def _guarded_lookup(host, port, *arguments, **keywords):
    if host is not None:
        _require_local((host, port))
    return _lookup(host, port, *arguments, **keywords)


socket.socket.connect = _guarded_connect
socket.socket.connect_ex = _guarded_connect_ex
socket.socket.sendto = _guarded_sendto
socket.getaddrinfo = _guarded_lookup
