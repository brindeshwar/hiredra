# Step 7 weight calibration: step7-2026-11-26

Owner-rated jobs only; no job text or employer names.

```
match thresholds: full credit from 0.81, partial from 0.55
rated jobs used: 17 tune + 9 holdout (4 left out: dropped by the filters since)
grid: 6300 combinations
current weights:  tune A 0.653, holdout A 0.769
best tune A 0.842; 5 combinations within 0.01
chosen (3 knobs changed): w_domain 1.0, w_gap 0.0, cover_bar 0.637, good-to-have 0.4, backend 0.5, data/ml 0.5, generic_swe 0.25
chosen:           tune A 0.837 (95 pairs), holdout A 0.731 (26 pairs), 90% interval 0.38-1.00
apply line: 1.321 (balanced accuracy on tune apply vs skip 1.00)
```

## Notes (owner decision, 2026-11-26)

- The holdout agreement (0.731, 26 pairs, 90% interval 0.38-1.00) is below the plan's 0.85 check-in bar. The owner
  chose to freeze these weights as version 1 and add 15 more ratings (`rate --add`), then calibrate again.
- Apply versus skip separates cleanly on both sets. The pairs the score gets wrong are "maybe" jobs ranked above
  "apply" ones; the owner's reasons for them (pay, years asked, interview timing) are not inputs to the formula.
- The same search under the proposed match thresholds (full credit by name or alias only, partial from 0.615) gave
  tune 0.795 and holdout 0.615. That change ("PR6") was dropped.
- The chosen gap weight is 0: on these ratings the gap term did not improve the ranking.

## Re-check after 15 more ratings (owner decision, 2026-11-28)

```
match thresholds: full credit from 0.81, partial from 0.55
rated jobs used: 27 tune + 14 holdout (4 left out: dropped by the filters since)
grid: 6300 combinations
current weights:  tune A 0.812, holdout A 0.727
best tune A 0.829; 4 combinations within 0.01
chosen (1 knobs changed): w_domain 1.0, w_gap 0.0, cover_bar 0.637, good-to-have 0.6, backend 0.5, data/ml 0.5, generic_swe 0.25
chosen:           tune A 0.825 (240 pairs), holdout A 0.727 (64 pairs), 90% interval 0.47-0.91
apply line: 1.326 (balanced accuracy on tune apply vs skip 0.94)
```

- The only change on offer (good-to-have weight 0.4 to 0.6) raised tune agreement slightly and left the holdout
  exactly where it was, so the owner kept version 1: nothing above was frozen.
- The holdout pairs the score gets wrong are again "maybe" jobs above "apply" ones for pay and years asked.
