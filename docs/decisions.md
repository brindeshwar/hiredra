# docs/decisions.md — Rejected approaches (do NOT re-introduce)

Project: **Hiredra**. These were considered and deliberately rejected, each for a specific reason.
If you (Claude Code) find yourself proposing one of these, that is a signal to STOP and ask the
owner. Do not "improve" the system by adding them back.

---

### 1. Neural network for scoring — REJECTED for v1
Reason: a NN needs hundreds–thousands of labeled examples; the owner has ~15 hand-rated jobs. A NN
on that data memorizes and generalizes to nothing. Even later, with a few hundred TABULAR rows and
~5 features, logistic regression or gradient-boosted trees beat/match a NN and stay auditable.
- Allowed instead: v1 = hand-calibrated weights. v2 (gated at ~150–200 real outcomes) = LEARN the
  weights, logistic regression first. NN only as a deliberate compared experiment, never the
  default, and only if it actually wins on the owner's data.

### 2. "Magnitude of the gap/ideal-vs-actual vector" as the whole score — REJECTED
Reason: a single magnitude measures distance-from-perfect, not fit. It ignores coverage (how much
you have) and domain, scales with job size, and ranks a perfect-match boring job above a target
role with one gap. A single scalar cannot carry three signals that pull in different directions.
- Allowed instead: gap magnitude survives as ONE TERM (`gap_distance`), computed over the GAP
  skills only, alongside separate coverage and domain terms.

### 3. Gap DIRECTION (using vector direction to encode domain) — REJECTED for this project
Reason: (a) the whole skill-matching field scores by magnitude/similarity, not gap-direction;
(b) direction/analogy arithmetic is documented as fragile on messy real inputs; (c) getting a
single gap-direction requires averaging missing skills, which is provably lossy and points to a
meaningless midpoint. Direction-as-meaning IS real, but it already lives per-skill in the embedding
space (that's what nearest-`mine` distance uses). Domain is handled better by the LLM bucket.
- Note: "per-item gap direction without averaging" is a legitimate KNOWN-HARD research direction,
  not something to build into this project.

### 4. Specialty-clustering (label a skill specialty by its position in the owner's cluster) — REJECTED
Reason: the owner's skills are scattered (a lonely outlier skill creates a false "specialty" zone
around itself — the "Unity problem"). Defining "my cluster" reliably is hard and noisy.
- Allowed instead: domain comes from the LLM classifying the whole JD into a bucket → trinary step
  {0, 0.5, 1}. The graph does gap-distance + coverage only, not domain.

### 5. Graph for ROLE-TYPE matching — REJECTED
Reason: role-type is a flat set-membership test (is this title in a target bucket?), not a
relationship-traversal problem. A graph here is over-engineering.
- Allowed instead: keyword/alias sets per bucket (code).

### 6. A separate vector store alongside the graph — REJECTED
Reason: the graph's node embeddings ARE the vectors. Standing up both is duplicate infrastructure.
- Allowed instead: ONE embedded index; similarity and gap-distance query it directly.

### 7. Pairwise-LLM ranking — DEFERRED (not built now)
Reason: extra LLM calls and less auditable than the formula. Not needed unless calibration shows
the formula genuinely can't separate the top slice.
- Allowed instead: rules + calibrated formula. Revisit pairwise only if calibration reveals a gap.

### 8. A separate domain analyzer / per-skill specialty labelling — REJECTED
Reason: domain is arithmetic over the LLM's single bucket label; no extra component needed.

### 9. Auto-submit / auto-send — REJECTED (hard rule)
Reason: re-opens the browser-automation CAPTCHA fight, is irreversible, and breaks
interview-defensibility (a model auto-answering legal/attestation fields). The human gate is the
selling point. See CLAUDE.md rule 1.

### 10. Keyword-based seniority filtering at SEARCH time — REJECTED
Reason: adding "fresher/junior/entry" to search terms backfires (senior JDs say "mentor junior
engineers"). Seniority must be judged from the JD BODY by code, after fetching.

### 11. n8n / visual workflow tool for Track 2 — REJECTED
Reason: Track 2 is a pure-Python system for full control and as a portfolio artifact. (n8n is a
separate existing system.)

### 12. Rewriting the CV renderer from scratch — REJECTED
Reason: it's deterministic plumbing (fill a template, print a PDF); rewriting teaches nothing and
adds no value. Reuse the career-ops Node renderer via subprocess and ADJUST it.

### 13. Top-N cap on jobs processed — REJECTED
Reason: the intent is to apply to ALL survivors in priority order, not discard. "10" is only the
human review-batch size.
- Allowed instead: filters cut hard, scoring ranks, expensive stages drain the full ranked list
  paced to budget.

### 14. Hand-guessed weights without calibration — REJECTED
Reason: this is exactly what made the prior system's scores untrustworthy. Weights MUST be
calibrated against the owner's hand-labeled jobs.

### 15. Copying wellfound_autoApply code (even modified) — REJECTED
Reason: it is UNLICENSED; a modified copy is still a derivative work. Study only; reimplement any
idea from scratch.

### 16. Storing a plaintext password (as naukri_update does) — REJECTED (hard rule)
Reason: security. Use a saved browser session; no password in any file.

### 17. Cycling API keys/accounts to bypass free-tier limits — FLAGGED, not approved
Reason: commonly against provider ToS (Groq/Gemini), risks account bans, hard to defend in
interview. Parked; weigh against tighter filters / paced draining / provider split before adopting.
Do not implement without explicit owner sign-off.

### 18. Building the graph before scoring can run / making scoring depend on unbuilt graph edges — REJECTED
Reason: scoring must be runnable and calibratable early. Scoring reads only `vector` + `mine`
(Phase A); edges (Phase B) come last and scoring never depends on them.
