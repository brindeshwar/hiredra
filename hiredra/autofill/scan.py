"""Read an application form before the human opens it, and sort every field (plan unit S9.2, simplified for the
end-to-end build: Greenhouse and Lever only).

- Greenhouse: the public questions API, ``GET boards-api.greenhouse.io/v1/boards/{board}/jobs/{id}?questions=true``.
- Lever: the job's apply page, read once as HTML (robots.txt allows it), each ``application-question`` taken as a
  label and its control. Lever adds an hCaptcha, so a Lever form always carries a CAPTCHA pause.
- Anything else: ``via: unsupported``. The job stays usable through its packet and the copy panel.

GET only, through the shared HTTP client (identifiable user agent, per-host limits, cooldowns). A 404 means the
posting is gone.

Every field gets one class, decided by keyword rules in code (rules 3 and 6; the model never decides):

- ``PAUSE``: anything the system must not answer -- work authorization, salary or notice period, relocation,
  demographics, legal consent, government IDs, how-did-you-hear, a CAPTCHA -- and **anything not recognised**;
- ``FACT``: a value from the owner's facts file (name, email, phone, links, education, current company);
- ``FILE``: the packet's résumé or cover letter;
- ``DRAFT``: a free-text question the profile can answer (why this role, describe a project...); drafted later.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from html.parser import HTMLParser

from hiredra.sourcing.http import HttpClient, HttpConfig

# ── classification ────────────────────────────────────────────────────────────
#: (category, pattern) checked first and in order: a match makes the field PAUSE.
PAUSE_RULES: tuple[tuple[str, re.Pattern], ...] = tuple(
    (cat, re.compile(pat, re.IGNORECASE))
    for cat, pat in (
        # A question that depends on another answer ("If you answered Yes above...") is the human's: the system
        # never answered the question it depends on.
        ("conditional", r"\bif (you )?(answered|selected|chose|ticked)|\bif (yes|no|so)\b|\bif applicable\b"),
        ("work_auth", r"authori[sz]|visa|sponsor|work.?permit|citizen|right to work|legally|eligib"),
        ("salary", r"salary|\bctc\b|compensation|remuneration|stipend|expected pay|pay expectation|current pay"),
        ("notice", r"notice period|start date|available to start|availability|joining|when can you"),
        ("relocation", r"relocat|on-?site|in.office|office\b|commute|hybrid|willing to work"),
        (
            "demographic",
            r"gender|\brace\b|ethnic|veteran|disabilit|pronoun|sexual|\bage\b|date of birth|religio|caste|hispanic",
        ),
        ("legal", r"privacy|consent|acknowledg|\bagree|terms|certif|attest|declar|policy|gdpr|accurate"),
        ("gov_id", r"aadhaa?r|\bpan\b|passport|\bssn\b|social security|national id"),
        ("judgment", r"how did you hear|hear about|\breferr|\bsource\b|anything else"),
    )
)
#: label pattern -> facts key, for single-line fields.
FACT_RULES: tuple[tuple[re.Pattern, str], ...] = tuple(
    (re.compile(pat, re.IGNORECASE), key)
    for pat, key in (
        (r"^first name|given name", "first_name"),
        (r"^last name|surname|family name", "last_name"),
        (r"preferred (first )?name", "preferred_name"),
        (r"^(full )?name\b|^your name", "full_name"),
        (r"e-?mail", "email"),
        (r"phone|mobile", "phone"),
        (r"linkedin|linkedln", "linkedin"),
        (r"github", "github"),
        (r"portfolio|website|personal site", "portfolio"),
        (r"current (company|employer|organi[sz]ation)|^company$|^org$", "current_company"),
        (r"current (title|role|position)", "current_title"),
        (r"^country", "country"),
        (r"^city|current location|^location", "location"),
        (r"university|school|college|institution", "institution"),
        (r"^degree|qualification", "degree"),
        (r"graduation (year|date)|year of graduation", "graduation_year"),
    )
)
_FILES = (
    (re.compile(r"resume|\bcv\b", re.IGNORECASE), "cv.pdf"),
    (re.compile(r"cover letter", re.IGNORECASE), "cover.pdf"),
)
_DRAFTABLE = re.compile(
    r"\bwhy\b|describe|tell us|experience (with|in)|interest|project|motivat|about (you|yourself)|excite|"
    r"accomplish|proud|challenge|what (would|do|can) you",
    re.IGNORECASE,
)


@dataclass
class Field:
    label: str
    key: str  # the form's own name for the control (what a filler targets)
    type: str  # text | textarea | email | tel | select | radio | checkbox | file
    required: bool = False
    options: list[str] = field(default_factory=list)
    cls: str = "PAUSE"
    facts_key: str | None = None
    file: str | None = None
    category: str | None = None


def classify(f: Field) -> Field:
    """Decide the field's class in place (and return it). Order: files, pause topics, facts, drafts, else pause."""
    label = " ".join(f.label.split())
    if f.type == "file":
        for pat, name in _FILES:
            if pat.search(label) or pat.search(f.key):
                f.cls, f.file = "FILE", name
                return f
        f.cls, f.category = "PAUSE", "unknown_file"
        return f
    for cat, pat in PAUSE_RULES:
        if pat.search(label):
            f.cls, f.category = "PAUSE", cat
            return f
    if f.type in ("text", "email", "tel"):
        for pat, key in FACT_RULES:
            if pat.search(label) or pat.search(f.key):
                f.cls, f.facts_key = "FACT", key
                return f
    if f.type in ("text", "textarea") and _DRAFTABLE.search(label):
        f.cls = "DRAFT"
        return f
    f.cls, f.category = "PAUSE", "unknown"
    return f


# ── the platforms ─────────────────────────────────────────────────────────────
_GH_TYPES = {"input_text": "text", "textarea": "textarea", "input_file": "file", "multi_value_single_select": "select",
             "multi_value_multi_select": "checkbox", "input_hidden": "hidden"}  # fmt: skip


def parse_greenhouse(data: dict) -> list[Field]:
    """The questions API's answer as fields. A question with a file and a text box (Resume/CV) keeps the file."""
    out: list[Field] = []
    blocks = [*(data.get("questions") or []), *(data.get("location_questions") or [])]
    for q in blocks:
        controls = [c for c in q.get("fields") or [] if _GH_TYPES.get(c.get("type")) != "hidden"]
        if not controls:
            continue
        chosen = next((c for c in controls if c.get("type") == "input_file"), controls[0])
        out.append(Field(
            label=str(q.get("label") or chosen.get("name") or ""), key=str(chosen.get("name") or ""),
            type=_GH_TYPES.get(chosen.get("type"), "text"), required=bool(q.get("required")),
            options=[str(v.get("label")) for v in chosen.get("values") or []],
        ))  # fmt: skip
    for block in ("compliance", "demographic_questions"):
        if data.get(block):
            out.append(Field(label=f"{block.replace('_', ' ')} questions", key=block, type="select", required=False))
    return out


class _LeverForm(HTMLParser):
    """Collects (label, control) pairs from Lever's ``<li class="application-question">`` blocks."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.fields: list[Field] = []
        self._in_q = self._in_label = False
        self._label: list[str] = []
        self._current: Field | None = None
        self.captcha = False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        cls = a.get("class") or ""
        if "h-captcha" in cls or "hcaptcha" in (a.get("src") or ""):
            self.captcha = True
        if tag == "li" and "application-question" in cls:
            self._in_q, self._label, self._current = True, [], None
        elif self._in_q and tag == "div" and "application-label" in cls:
            self._in_label = True
        elif self._in_q and tag in ("input", "textarea", "select") and a.get("type") != "hidden" and a.get("name"):
            kind = "textarea" if tag == "textarea" else "select" if tag == "select" else (a.get("type") or "text")
            existing = next((f for f in self.fields if f.key == a["name"]), None)
            if existing is None:  # radio buttons and checkboxes share one name: one field with options
                existing = Field(label="", key=a["name"], type=kind, required="required" in a)
                self.fields.append(existing)
            if kind in ("radio", "checkbox") and a.get("value"):
                existing.options.append(a["value"])

    def handle_endtag(self, tag):
        if tag == "div" and self._in_label:
            self._in_label = False
        if tag == "li" and self._in_q:
            label = " ".join("".join(self._label).replace("✱", "").split())
            for f in self.fields:
                if not f.label:
                    f.label = label
            self._in_q = False

    def handle_data(self, data):
        if self._in_label:
            self._label.append(data)


def parse_lever(html: str) -> tuple[list[Field], bool]:
    """The apply page's fields, and whether it carries a CAPTCHA. The résumé upload sits outside the question blocks."""
    p = _LeverForm()
    p.feed(html)
    fields = p.fields
    if 'name="resume"' in html and not any(f.key == "resume" for f in fields):
        fields.insert(0, Field(label="Resume/CV", key="resume", type="file", required=True))
    seen, unique = set(), []
    for f in fields:
        if f.key not in seen:
            seen.add(f.key)
            unique.append(f)
    return unique, p.captcha or "h-captcha" in html


# ── one job ───────────────────────────────────────────────────────────────────
_GH_HOST = re.compile(r"(?:job-boards(?:\.eu)?|boards)\.greenhouse\.io/([^/?#]+)/jobs/(\d+)")


def scan_job(job: dict, http: HttpClient | None = None) -> dict:
    """The form scan for one job (``form_scan.json``'s content). Never raises for a dead posting: ``dead`` is set."""
    http = http or HttpClient(HttpConfig())
    source, url = job.get("source"), job.get("apply_url") or job.get("url") or ""
    scan = {"job_id": job["job_id"], "ats": source, "via": "unsupported", "url": url, "dead": False, "captcha": False,
            "scanned_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "fields": []}  # fmt: skip
    try:
        if source == "greenhouse":
            m = _GH_HOST.search(job.get("url") or "")
            board = job.get("company_slug") or (m.group(1) if m else None)
            job_id = job.get("external_id") or (m.group(2) if m else None)
            data = http.get_json(f"https://boards-api.greenhouse.io/v1/boards/{board}/jobs/{job_id}?questions=true")
            fields = parse_greenhouse(data)
            scan["via"] = "api"
        elif source == "lever":
            apply = url if url.rstrip("/").endswith("/apply") else url.rstrip("/") + "/apply"
            fields, scan["captcha"] = parse_lever(http.get_text(apply))
            scan["via"], scan["url"] = "static", apply
        else:
            return scan
    except Exception as exc:  # a 404 or a gone board: the posting is closed
        if getattr(exc, "status", None) in (404, 410) or "404" in str(exc):
            scan["dead"] = True
            return scan
        raise
    fields = [classify(f) for f in fields]
    if scan["captcha"]:
        fields.append(Field(label="CAPTCHA (solved by you when you submit)", key="captcha", type="captcha",
                            cls="PAUSE", category="captcha"))  # fmt: skip
    scan["fields"] = [asdict(f) for f in fields]
    return scan
