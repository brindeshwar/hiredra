"""One model call, the way every writing step makes it.

- Paced under ``requests_per_minute``.
- Counted on the NVIDIA daily meter (``api_usage``, provider ``nvidia``), shared with the extraction fallback: a
  request counts once NVIDIA accepts it (the owner's rule, 2026-11-11), and the meter stops a run at the guard.
- Passing failures (429, 5xx, a dropped connection) are retried until they clear: 1, 2, 4, 8 s doubling plus
  jitter, capped at 120 s (the owner's rule). Only the daily guard stops a run.
- Every call is written to ``llm_calls`` (stage, job, ok, error class, sizes, time): never the text itself.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from pydantic import BaseModel, ValidationError
from tenacity import Retrying, retry_if_exception, stop_never

from hiredra.quota import DailyUsage, classify_failure, jittered_wait
from hiredra.resume.settings import ResumeConfig
from hiredra.storage import record_llm_call


class Caller:
    """``call(prompt, schema, stage=..., job_id=...) -> schema instance | None``."""

    def __init__(
        self,
        conn,
        cfg: ResumeConfig,
        *,
        chain_factory: Callable[[type[BaseModel]], object] | None = None,
        usage: DailyUsage | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.conn, self.cfg = conn, cfg
        self.usage = usage
        self._factory = chain_factory or self._nvidia
        self._chains: dict[type, object] = {}
        self._sleep, self._clock = sleep, clock
        self._gap, self._next = 60.0 / max(1, cfg.requests_per_minute), None
        self.calls = 0

    def _nvidia(self, schema: type[BaseModel]):
        from hiredra.nvidia import NvidiaChain

        return NvidiaChain(
            self.cfg.model, schema, base_url=self.cfg.base_url, timeout=self.cfg.timeout_seconds,
            max_tokens=self.cfg.max_tokens,
        )  # fmt: skip

    def call(self, prompt: str, schema: type[BaseModel], *, stage: str, job_id: str | None = None):
        chain = self._chains.get(schema)
        if chain is None:
            chain = self._chains[schema] = self._factory(schema)

        def once():
            if self.usage is not None:
                self.usage.reserve(1)
            if self._next is not None and self._clock() < self._next:
                self._sleep(self._next - self._clock())
            self._next = self._clock() + self._gap
            started = self._clock()
            try:
                result = chain.invoke(prompt)
            except ValidationError:  # the answer broke the schema: count it as no answer (the caller asks again)
                if self.usage is not None:
                    self.usage.record(1)
                record_llm_call(self.conn, model=self.cfg.model, stage=stage, job_id=job_id, ok=False,
                                error_class="schema_error", in_chars=len(prompt),
                                latency_ms=int((self._clock() - started) * 1000))  # fmt: skip
                return None
            except Exception as exc:
                record_llm_call(self.conn, model=self.cfg.model, stage=stage, job_id=job_id, ok=False,
                                error_class=classify_failure(exc) or type(exc).__name__, in_chars=len(prompt),
                                latency_ms=int((self._clock() - started) * 1000))  # fmt: skip
                raise
            if self.usage is not None:
                self.usage.record(1)  # accepted by NVIDIA: it counts
            self.calls += 1
            out = result.model_dump_json() if result is not None else ""
            record_llm_call(self.conn, model=self.cfg.model, stage=stage, job_id=job_id, ok=result is not None,
                            error_class=None if result is not None else "empty", in_chars=len(prompt),
                            out_chars=len(out), latency_ms=int((self._clock() - started) * 1000))  # fmt: skip
            return result

        retrying = Retrying(
            retry=retry_if_exception(lambda exc: classify_failure(exc) == "transient"),
            stop=stop_never,
            wait=jittered_wait,
            sleep=self._sleep,
            reraise=True,
        )
        return retrying(once)
