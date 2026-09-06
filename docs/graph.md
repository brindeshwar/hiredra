# docs/graph.md — Knowledge graph (full detail)

Project: **Hiredra**. Read this before building the graph (build step 4) and the fit mechanic.
This is the full version of `CLAUDE.md` section 5.

The graph is a **capability graph** bounded to the owner's skills + skills actually seen in real
JDs. It is NEVER a full tech ontology. Its scope boundary is a hard rule: if it starts trying to
model all of technology, stop — that version dies of maintenance.

---

## 1. What the graph is for
- Scoring (Phase A): supply the per-skill fit (nearest-`mine` distance) used by coverage and
  gap_distance. Scoring reads only `vector` and `mine`.
- Later features (Phase B): "what should I learn next", eligibility/gap analysis, auto-updating
  portfolio. These use edges.

---

## 2. The Skill object (schema — final now, fill over time)

```
Skill:
  name     : str              # canonical name, e.g. "fastapi"
  aliases  : [str]            # ["Fast API", "fastapi", "FastAPI framework"]
  mine     : bool             # true if the owner has it (from the master profile), else market
  level    : str | null       # only if mine: production | built-with | learned | interview-ready
  vector   : [float]          # the embedding; this IS the vector used for similarity/gap
  edges    : [ {type, target} ]  # Phase B only; stays EMPTY through Phase A
```

- The `edges` field EXISTS from day one but stays empty in Phase A. This makes Phase B a matter of
  filling a field, not changing the schema (no migration, no rework).
- `vector` doubles as the similarity index — there is ONE embedded index, not a graph plus a
  separate vector store.

---

## 3. Two build phases

### Phase A (build now) — embeddings only
Everything except edges. Delivers scoring's per-skill fit. "Proven" when the specialty/gap results
look sensible on real jobs AND weight calibration passes (docs/scoring.md 5b).

### Phase B (build last) — edges
An LLM pass fills `edges`. Gated on Phase A + weight calibration proving out. Unlocks learning-path
/ eligibility / portfolio. Scoring never reads edges, so adding them touches nothing in scoring.

Edge types (Phase B): `used-in`, `component-of`, `related-to`, `prerequisite-for`.

---

## 4. How the graph is built (fill order)

1. **Backfill scrape (once, refreshed periodically):** a WIDE, UNFILTERED scrape (all seniorities,
   all roles) whose only purpose is to harvest skill vocabulary. This is SEPARATE from scoring
   runs — do not pollute scoring with these junk jobs; only their skills are used.
2. **Extract skills** from the scrape (LLM).
3. **Normalize / dedup** to canonical names + aliases. ~thousands of raw mentions collapse to
   ~300–500 distinct canonical skills. These become `market` nodes (`mine: false`).
4. **Embed** each canonical skill → `vector`. Place in the single index.
5. **Overlay the master profile** → set `mine: true` and `level` on the owner's skills
   (matching described in section 5).
6. **(Phase B, later)** LLM draws edges.

New skills seen during normal scoring runs are added incrementally between runs (section 6).

---

## 5. Master profile overlay (matching owner's skills to market skills)

Goal: when overlaying the owner's skills, correctly unify each with the matching market node
(so there aren't two "langchain" nodes). Two passes:

1. **Exact canonical-name match** — catches everything normalization already unified (~80%). Fast,
   certain.
2. **High-threshold vector check for the leftovers** — for an owner skill that didn't exact-match,
   embed it and find the nearest market node; if similarity ≥ 0.9, treat as the SAME concept under
   a different name (e.g. "LangChain agents" ≈ "agentic frameworks") and unify/link.
   - NEVER auto-merge merely-similar skills (≥0.7): that would wrongly fuse Docker and Kubernetes.
     0.9+ only.

Note: normalization solves surface forms (casing/spacing/aliases). The 0.9 vector pass catches
genuine synonyms (different words, same meaning). Both are needed.

---

## 6. Freeze rule + new-skill handling (important for stability)

- The graph is **FROZEN during a scoring run** so scores stay stable within a run.
- A new skill (not yet a node) appearing mid-run is still scored correctly: embed it fresh and
  compute distance to `mine` skills (the fit calc never needs it to be a stored node).
- New skills are **queued and added between runs**, not mid-run. (Chosen so the freeze holds and
  scoring/ingestion stay decoupled — scoring never waits on ingestion.)
- New skills are RARE right after a backfill (the backfill already harvested the market
  vocabulary). Keep them rare by refreshing the backfill periodically (~quarterly), since the
  market drifts (new frameworks appear).

---

## 7. Source-of-truth rule (do not violate)
- The owner edits the **master profile file**. The graph's `mine` nodes are REBUILT from it.
- Never edit capability directly in the graph. One source of truth = the file; the graph is a
  derived view. This prevents two-sources-of-truth drift.

---

## 8. The master profile file (spec)
- SEPARATE from the resume. The resume is a trimmed view; the master profile has EVERYTHING the
  owner has worked on or knows, no length limit.
- Structured, tagged entries (not prose) so scoring and resume-generation can query it.
- Each entry carries a `level`: production / built-with / learned / interview-ready.
  - `interview-ready` skills COUNT toward fit but are flagged "do not claim as production
    experience on the resume." This keeps the interview-defensibility rule automatic.
- This file feeds: the graph's `mine` nodes, the fit calc, and the resume generator.

---

## 9. Scope boundary (hard rule)
- Nodes = owner's skills + skills seen in real JDs. Nothing else.
- No modelling of the entire tech universe. Bounded to what is useful.
