"""Pipeline stages started from the app, each as its own process (the same commands as the command line).

A run is a child process of the API; its output is kept in memory (the last 200 lines) for the app to show. One run
per stage at a time: a second start of the same stage is refused (the API answers 409). A stage that another
process holds (its run lock) exits with code 4 and says "NOT RUN"; that shows here as a failed run with its log.
Nothing here submits anything: these are the pipeline's own read and prepare steps.
"""

from __future__ import annotations

import subprocess
import sys
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: stage -> the command after ``python``. ``{top}`` is filled from the request (packets only).
STAGES: dict[str, list[str]] = {
    "filter": ["-m", "hiredra.filter"],
    "extraction": ["-m", "hiredra.extraction"],
    "sync": ["-m", "hiredra.graph", "sync"],
    "describe": ["-m", "hiredra.graph.describe"],
    "scoring": ["-m", "hiredra.scoring", "run"],
    "audit": ["-m", "hiredra.graph", "audit"],
    "packets": ["-m", "hiredra.resume", "packets", "--top", "{top}"],
}


@dataclass
class Run:
    id: str
    stage: str
    started_at: str
    process: subprocess.Popen | None = None
    log: deque = field(default_factory=lambda: deque(maxlen=200))
    exit_code: int | None = None
    finished_at: str | None = None

    @property
    def state(self) -> str:
        if self.exit_code is None:
            return "running"
        return "ok" if self.exit_code == 0 else "failed"

    def view(self) -> dict:
        return {"id": self.id, "stage": self.stage, "state": self.state, "exit_code": self.exit_code,
                "started_at": self.started_at, "finished_at": self.finished_at, "log": list(self.log)}  # fmt: skip


class Runs:
    """The runs this API has started."""

    def __init__(
        self, stages: dict[str, list[str]] | None = None, python: str = sys.executable, cwd: Path = _REPO_ROOT
    ):
        self.stages = stages or STAGES
        self.python, self.cwd = python, cwd
        self._runs: dict[str, Run] = {}
        self._lock = threading.Lock()

    def active(self, stage: str) -> Run | None:
        return next((r for r in self._runs.values() if r.stage == stage and r.state == "running"), None)

    def start(self, stage: str, **params) -> Run:
        """Start ``stage``. KeyError for an unknown stage, RuntimeError if that stage is already running here."""
        args = [a.format(**{k: str(v) for k, v in params.items()}) for a in self.stages[stage]]
        with self._lock:
            if self.active(stage):
                raise RuntimeError(f"{stage} is already running")
            run = Run(id=uuid.uuid4().hex[:12], stage=stage, started_at=_now())
            run.process = subprocess.Popen(
                [self.python, *args], cwd=self.cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", env=_child_env(),
            )  # fmt: skip
            self._runs[run.id] = run
        threading.Thread(target=self._follow, args=(run,), daemon=True).start()
        return run

    def _follow(self, run: Run) -> None:
        for line in run.process.stdout:
            run.log.append(line.rstrip())
        run.exit_code = run.process.wait()
        run.finished_at = _now()

    def get(self, run_id: str) -> Run | None:
        return self._runs.get(run_id)

    def all(self) -> list[Run]:
        return sorted(self._runs.values(), key=lambda r: r.started_at, reverse=True)

    def stop_all(self) -> None:
        for r in self._runs.values():
            if r.state == "running" and r.process is not None:
                r.process.terminate()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _child_env() -> dict:
    import os

    env = dict(os.environ)
    env.pop("HIREDRA_API_TOKEN", None)  # a stage never needs the app's token
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    return env
