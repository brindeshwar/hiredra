"""Turn a form scan into answers: what each field gets, ready for the copy panel and the filler.

- ``FACT``: the value from the owner's facts file (a fact the file does not have becomes a ``PAUSE``, never a guess);
- ``FILE``: the path of the packet's résumé or cover letter PDF;
- ``DRAFT``: one model call per form for all its draft questions (NVIDIA, the same caller as the packets), each
  answer checked by the claim gate. One that fails becomes a ``PAUSE``: the human answers it.
- ``PAUSE``: no value, ever. Listed for the human with its category.

The result is ``answers.json`` in the packet folder, and the job's attention items (labels only, never values).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

from hiredra.resume.claims import Gate
from hiredra.resume.source import ProfileText

PROMPT_VERSION = "answers-1"


class DraftAnswer(BaseModel):
    label: str
    answer: Annotated[str, StringConstraints(min_length=40, max_length=1200)]


class DraftAnswers(BaseModel):
    answers: list[DraftAnswer] = Field(min_length=1)


def build_prompt(job: dict, profile: ProfileText, questions: list[str], cv_summary: str) -> str:
    entries = "\n\n".join(f"### {e.heading}\n" + "\n".join(f"- {b}" for b in e.bullets) for e in profile.entries)
    asked = "\n".join(f"- {q}" for q in questions)
    return (
        "Answer these application questions for the candidate, in the first person, 2 to 4 sentences each, using "
        "only facts from the profile below. Never add a skill, number, employer or result the profile does not "
        "state. No em dashes. Copy each question exactly as the label.\n\n"
        f"JOB: {job.get('title') or ''} at {job.get('company') or ''}\n"
        f"{(job.get('description') or '')[:3000]}\n\n"
        f"CANDIDATE SUMMARY: {cv_summary}\n\nPROFILE:\n{entries}\n\nQUESTIONS:\n{asked}\n"
    )


def build_answers(scan: dict, facts: dict[str, str], *, job: dict, profile: ProfileText, gate: Gate, caller=None,
                  cv_summary: str = "", packet_dir=None) -> dict:  # fmt: skip
    """``answers.json``'s content. ``caller`` may be None: draft questions then become PAUSE (no model call)."""
    fields = []
    for f in scan.get("fields") or []:
        out = {"label": f["label"], "key": f["key"], "type": f["type"], "class": f["cls"], "value": None,
               "file": None, "category": f.get("category"), "options": f.get("options") or []}  # fmt: skip
        if f["cls"] == "FACT":
            value = facts.get(f.get("facts_key") or "")
            if f.get("facts_key") == "location" and not value:
                value = ", ".join(x for x in (facts.get("city"), facts.get("country")) if x) or None
            if value:
                out["value"] = value
            else:
                out["class"], out["category"] = "PAUSE", "missing_fact"
        elif f["cls"] == "FILE":
            out["file"] = f.get("file")
            out["value"] = str(packet_dir / f["file"]) if packet_dir is not None else f.get("file")
        fields.append(out)

    drafts = [o for o in fields if o["class"] == "DRAFT"]
    if drafts and caller is None:
        for o in drafts:
            o["class"], o["category"] = "PAUSE", "not_drafted"
    elif drafts:
        context = f"{job.get('title') or ''} {job.get('company') or ''}"
        result = caller.call(build_prompt(job, profile, [o["label"] for o in drafts], cv_summary), DraftAnswers,
                             stage="answers", job_id=job["job_id"])  # fmt: skip
        got = {" ".join(a.label.split()).casefold(): a.answer for a in (result.answers if result else [])}
        for o in drafts:
            text = got.get(" ".join(o["label"].split()).casefold())
            if text and not gate.check(f"answer:{o['label'][:40]}", text, context=context):
                o["value"] = " ".join(text.split())
            else:
                o["class"], o["category"] = "PAUSE", "draft_failed" if text else "not_drafted"

    return {
        "job_id": scan["job_id"],
        "via": scan.get("via"),
        "url": scan.get("url"),
        "dead": scan.get("dead", False),
        "made_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "prompt_version": PROMPT_VERSION,
        "fields": fields,
    }
