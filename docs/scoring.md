# docs/scoring.md — Scoring engine (full detail)

Project: **Hiredra**. Read this before building any scoring slice (build steps 3–7). This is the
full version of the summary in `CLAUDE.md` section 4. The core principle governs everything here:

> **The LLM only reads and normalizes messy text. Code computes every number and makes every
> decision. The model must never emit a score or a ranking.**

---

## 1. The funnel

Jobs pass through cheap deterministic filters first; the LLM is only spent on survivors. Each
stage removes a different kind of noise.

### Stage 1 — Seniority gate (CODE)
- Reads the JD **body**, not just the title. Titles lie ("Software Engineer" with "5+ years
  required" inside; senior JDs that say "mentor junior engineers").
- Mechanism: regex to extract required years ("3+ years", "minimum 5 years", "2-4 yrs") + a small
  anti-signal phrase list ("mentor junior", "lead a team", "own the roadmap", "principal",
  "staff", "architect", etc.).
- Keep: fresher / 0–1 yr / roles open to 0–2 yr. **Interns and trainees are INCLUDED.**
- Drop anything above ~2 yr.
- Seniority is a GATE ONLY. It is not a term in the score. Everything that reaches scoring is
  already seniority-valid, so scoring it again would score a constant.

### Stage 2 — Role-type filter (CODE)
- Keep only jobs whose title/JD belongs to a target bucket: backend / agent / LLM / data.
- Mechanism: a keyword/alias set per bucket. Flat set-membership test. **Not a graph** — this is a
  coarse yes/no lookup, not a relationship problem.
- This is the COARSE domain cut (drops wrong-role jobs). The FINE domain distinction (how
  on-target a surviving job is) happens in scoring via the domain step (section 3).

### Stage 3 — Legitimacy (LLM, bundled into extraction)
- Flag ghost jobs / spam / vague postings. One boolean/flag, produced inside the Stage-4 call so
  it costs no extra request.

### Stage 4 — Skill extraction (LLM, extract + normalize only)
- Input: the JD text. Output: strict structured JSON.
- Extract the required skills, **normalize each to a canonical name** (map "Fast API", "fastapi",
  "FastAPI framework" → `fastapi`).
- Tag each skill **required** vs **good-to-have**.
- Classify the whole job into exactly ONE domain bucket from a fixed closed list
  (agent/LLM, backend, data, ML, generic-SWE, other). Closed list — the model picks, never invents
  a label.
- Include the legitimacy flag (Stage 3) in the same response.
- **No scoring in this stage.** The model extracts and labels; it does not rate.
- Use strict/structured output (JSON schema) via direct HTTP to the model, not a loose prompt.

### Stage 5 — Score (CODE)
- Computes the formula (section 2) from the extracted evidence + the graph's per-skill fit.

---

## 2. The formula (CODE)

```
score = w_domain * domain_fit
      + w_skills * coverage
      - w_gap   * gap_distance
      + legitimacy_flag
```

All weights are set by calibration (section 5). Three inputs plus legitimacy. Each is defined
below.

### domain_fit — trinary step {0, 0.5, 1}
- Comes from the LLM's domain bucket (Stage 4), mapped via a config the owner controls:
  - agent/LLM → 1.0
  - backend → 0.5
  - data → 0.5
  - ML → 0.5
  - generic-SWE / other → 0.0
- Trinary (not binary) so a backend role (owner's second lane) sits above an irrelevant role but
  below the bullseye. These step-values are a starting point and are tuned in weight calibration.
- This is arithmetic over the bucket label, NOT a separate domain analyzer, and NOT computed from
  the graph.

### coverage — "how many required skills do I clear" (count, not size)
From the per-skill fit vector (section 4), threshold each skill to pass/fail, then average.
```
fit vector      = {1.0, 1.0, 1.0, 0.82, 0.1}
thresholded     = {1,   1,   1,   1,    0 }     # ≥ COVER_BAR (e.g. 0.6) → 1 else 0
coverage        = mean(thresholded) = 4/5 = 0.8
```

### gap_distance — "how big are the holes" (size, not count)
From the same fit vector: zero out the skills that pass, take the magnitude of the shortfalls on
the ones that don't.
```
fit vector      = {1.0, 1.0, 1.0, 0.82, 0.1}
shortfall       = {0,   0,   0,   0.18, 0.9}    # (1 - fit) but only for skills below the bar
gap_distance    = sqrt(0.18^2 + 0.9^2) ≈ 0.92
```
Coverage and gap_distance are deliberately non-redundant: coverage is a **count** (threshold then
average), gap_distance is a **size** (zero the wins, then magnitude). They extract opposite halves
of the fit vector. Two jobs with equal coverage but different hole sizes must rank differently —
that is why both exist.

### good-to-have factor `x`
- Good-to-have skills (tagged in Stage 4) are down-weighted by a factor `x` (start ~0.4) in BOTH
  coverage and gap_distance. A missing good-to-have barely dents either number; a missing required
  skill hurts fully.

### legitimacy_flag
- A small adjustment (or a gate) from Stage 3. A failed-legitimacy job is flagged and routed to
  review rather than trusted (see section 6).

---

## 3. Where domain is handled (do not rebuild this)
- COARSE: Stage-2 role filter drops wrong-role jobs.
- FINE: the domain step in the formula separates on-target from adjacent among survivors.
- There is NO specialty-clustering, NO per-skill specialty labelling by graph position, and NO
  separate domain model. Domain comes from the LLM bucket. (See docs/decisions.md.)

---

## 4. Per-skill fit mechanic (needs the graph — see docs/graph.md)

For each required skill in a JD, produce a fit value in [0, 1]:

1. **Embed the JD skill fresh** (turn its canonical name into a vector). Never require it to
   already be a node in the graph.
2. **Find the single nearest `mine` skill** by cosine similarity. Call it `s`.
3. **Map `s` to fit via two thresholds** (`HAVE`, `ADJ` — set by threshold calibration, section 5):
   ```
   s >= HAVE            -> fit = 1.0     # you have it (exact / near-identical)
   ADJ <= s < HAVE      -> fit = s       # adjacent; credit scales with closeness (keeps gap size)
   s < ADJ              -> fit = 0.0     # missing; low similarity is noise, floored to 0
   ```

Worked example — JD wants `[python, fastapi, postgresql, kubernetes, kafka]`, owner has
python, fastapi, postgresql, langchain, mcp, docker, aws:
```
skill        nearest(mine)   s      fit     bucket
python       python          1.00   1.00    have
fastapi      fastapi         1.00   1.00    have
postgresql   postgresql      1.00   1.00    have
kubernetes   docker          0.82   0.82    adjacent
kafka        aws             0.31   0.00    missing
fit vector = {1.0, 1.0, 1.0, 0.82, 0.0}
```
This fit vector feeds coverage and gap_distance (section 2).

Design rules for this mechanic:
- Adjacent fit = `s` itself (not a flat 0.5) — preserves per-skill gap size.
- Missing = hard 0 (not `s`) — below the floor, similarity is noise; do not leak phantom credit.
- The fit calc depends only on distance to `mine` skills, so a brand-new skill (not yet a node)
  still works: embed it and compare. (See docs/graph.md freeze/edge-case rules.)

---

## 5. Calibration (TWO calibrations, both required, in order)

Backing: setting cosine thresholds by an F1-maximizing sweep over a hand-labeled set is the
standard, published method for embedding-based classification (Reimers-style PR/F1 sweep). Results
are typically robust to the exact cutoff — aim for the right neighborhood, not a magic decimal.

### 5a. Threshold calibration (makes MATCHING accurate) — do first
Sets `HAVE` and `ADJ` for the chosen embedding model.
1. Pick and freeze the embedding model first (nothing here is meaningful without it).
2. Build ~50 hand-labeled skill pairs, balanced, e.g. `(kubernetes, docker) = related`,
   `(kafka, fastapi) = unrelated`, `(fastapi, "Fast API") = same`.
3. Compute cosine similarity for each pair.
4. Sweep candidate thresholds ~0.30 → ~0.95 in 0.05 steps; compute precision/recall/F1 against the
   labels; pick the threshold that maximizes F1. Use the "same" tier to set `HAVE`, the
   "related vs unrelated" boundary to set `ADJ`.
5. Validate on ~10–20 held-out pairs.

### 5b. Weight calibration (makes the SCORE trustworthy) — do after scoring is built
Sets `w_domain`, `w_skills`, `w_gap`, the `COVER_BAR`, and the apply/skip line.
1. Hand-rate ~15 real jobs: apply / maybe / skip.
2. Run them through the scorer.
3. Tune the weights + step-values until the formula's RANKING matches the owner's ratings
   (target: agrees ~9/10 on held-out jobs).
4. This is the ONLY way the owner's taste enters the math. Scoring is not "done" until it passes.

### Freezing & re-calibration
- Weights/thresholds are FROZEN between calibrations so scores are comparable across runs.
- Re-calibrate deliberately/periodically (e.g. quarterly, re-rate some jobs), NOT every run.
  Auto-tuning every run makes scores unstable (the same job scoring differently week to week).

---

## 6. Confidence flag (thin JDs)
- If extraction yields too little real signal (few/vague skills, thin JD), flag the job.
- Basis: "did extraction find enough real skills/requirements to score confidently", not just raw
  length (dense short JDs are fine; long fluffy ones aren't).
- Flagged jobs go to the REVIEW list — their score is not trusted blindly.

---

## 7. Data logging for future learning (start day one)
The ledger stores, per job:
- Scoring FEATURES: domain bucket, coverage, gap_distance, the fit vector, extracted skills.
- The eventual OUTCOME: applied / reply / interview / rejection / silence (filled in later as it
  arrives — the learning loop updates it).
This (features → outcome) is training data for a future v2 that LEARNS the weights from real
outcomes. Model choice deferred; **logistic regression first, NOT a neural net by default**
(right-sized for a few hundred tabular rows). Gate the v2 at ~150–200 labeled outcomes. Only the
data collection is committed now.

---

## 8. Scale interaction (see CLAUDE.md 3.5)
- Score is cheap (code) — run it on the full filtered pool (hundreds–thousands), batched and
  rate-limit-aware for the Stage-4 LLM calls.
- No top-N cut. Scoring RANKS; the expensive downstream stages drain the full ranked list in
  priority order, paced to the token budget.

---

## 9. Deferred (ask owner when reached)
- Streaming vs buffered apply gate (decide by experiment: start simple immediate-process; add a
  shallow best-first buffer only if runs show rate-limit/ordering pain).
- Provider/token split (which agent runs on Groq vs Gemini).
- Embedding model choice.
- v2 learner model choice.
