"""Start the sidecar API for the desktop app.

    HIREDRA_API_TOKEN=<token> python -m hiredra.api [--port 0]

Binds 127.0.0.1 only (never another interface). Port 0 picks a free port. When it is listening it prints exactly one
line, ``{"ready": true, "port": N}``, which the app reads. The token comes from the environment (the app generates
one per launch); without one the API refuses to start.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys

import uvicorn

from hiredra.api.app import create_app

HOST = "127.0.0.1"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m hiredra.api", description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", type=int, default=0, help="0 = any free port")
    ap.add_argument("--db", default=None, help="database URL (default: HIREDRA_DATABASE_URL)")
    args = ap.parse_args(argv)
    token = os.environ.get("HIREDRA_API_TOKEN", "")
    if len(token) < 16:
        print("NOT STARTED: set HIREDRA_API_TOKEN (16+ characters); the desktop app does this itself", file=sys.stderr)
        return 2
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((HOST, args.port))
    port = sock.getsockname()[1]
    app = create_app(token=token, db_url=args.db)
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning", access_log=False))
    print(json.dumps({"ready": True, "port": port}), flush=True)
    server.run(sockets=[sock])
    return 0


if __name__ == "__main__":
    sys.exit(main())
