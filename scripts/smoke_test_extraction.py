"""Live smoke test for hiredra.extraction -- NOT a pytest test.

Deliberately kept outside tests/ (not collected by pytest, same reasoning as
discovery/test_harvest.py): this hits the real Gemini API, needs a real
GEMINI_API_KEY in .env, and burns real free-tier quota. Run it manually,
once, to confirm the two real unknowns from the build plan:

1. `ChatGoogleGenerativeAI.with_structured_output` actually sends Gemini a
   schema with no `$ref`/`$defs` for the nested `skills: list[SkillItem]`
   shape (confirmed offline via direct inspection of langchain_google_genai's
   dereference_refs + _format_json_schema_to_gapic pipeline -- this is the
   live end-to-end confirmation).
2. `reasoning_effort="minimal"` doesn't break structured-output parsing (the
   thinking + forced-schema-decoding coexistence the build plan flagged as
   genuinely unknown).

Feeds two REAL JDs pulled straight from data/hiredra.db (not synthetic
text): one with an explicit required/nice-to-have split (WebSenor), one
requiring inference from framing rather than a labeled section (an OpenAI
"you might thrive in this role if you" posting) -- see the characterization
pass, sec 2.

    python scripts/smoke_test_extraction.py
"""

from __future__ import annotations

import json
import sqlite3

from hiredra.extraction import (
    build_extraction_chain,
    build_extraction_graph,
    load_extraction_config,
    preprocess,
)
from hiredra.storage import DEFAULT_DB_PATH

JOB_IDS = (
    "jobspy-indeed:in-af3ed2fa83dcc5d5",  # WebSenor -- explicit Required/Preferred split
    "ashby:c442eab0-b8b4-40f6-b0f2-0fc91aea294a",  # OpenAI -- inferred, no labeled split
)


def main() -> int:
    config = load_extraction_config()
    conn = sqlite3.connect(str(DEFAULT_DB_PATH))
    conn.row_factory = sqlite3.Row

    print("Building extraction chain (gemini-3.5-flash-lite, reasoning_effort='minimal')...")
    chain = build_extraction_chain()
    graph = build_extraction_graph(chain)
    print("Chain + graph built OK.\n")

    for job_id in JOB_IDS:
        row = conn.execute(
            "SELECT job_id, title, company, description FROM jobs WHERE job_id = ?",
            (job_id,),
        ).fetchone()
        if row is None:
            print(f"!! job_id {job_id!r} not found in {DEFAULT_DB_PATH} -- skipping")
            continue

        clean_text, strip_log = preprocess(row["description"] or "", row["company"], config)
        print("=" * 78)
        print(f"job_id={row['job_id']}  company={row['company']!r}  title={row['title']!r}")
        print(f"raw_len={len(row['description'] or '')}  clean_len={len(clean_text)}  strips={strip_log}")
        print("-" * 78)

        result = graph.invoke({"jd_text": clean_text})["result"]
        print("RAW RESULT:")
        print(json.dumps(result.model_dump(), indent=2))
        print()

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
