"""A small local counter for Google's per-DAY embedding limit (build 4.3 fix).

Why it exists: Google counts every input inside a batch as one request, and the
free tier allows 1,000 requests per day (resetting at midnight Pacific). The
first real index build ran into that cap with no warning. This counter lets a
run stop BEFORE the cap instead of after a rejected request, and remembers a cap
that Google has already reported, so a second run the same day refuses at once
without making any API call.

It is ADVISORY. Google is the source of truth: if this file is missing, wrong,
or from another day, the worst case is one rejected request, which the embedder
already handles. It lives in ``data/`` (git-ignored) as one tiny JSON file. The code moved to
``hiredra/quota.py`` (M0.1) so extraction shares it; this module keeps the embedding
counter's path and re-exports the names.
"""

from __future__ import annotations

from pathlib import Path

from hiredra.quota import (  # noqa: F401 -- moved to hiredra/quota.py (M0.1); re-exported unchanged
    IST,
    PACIFIC,
    DailyBudgetReached,
    DailyUsage,
    next_reset,
    pacific_day,
    reset_text,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_USAGE_PATH = _REPO_ROOT / "data" / "embed_usage.json"
