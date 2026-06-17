# Conv-SPROUT Phase 2 — conv-sprout-long-motifs

- **Dataset:** motifs  |  **Seeds:** 5  |  **Steps:** 60000  |  **Baseline:** fixed-hand-k6
- **Head:** sparse phasic (w32-sparse economy), conv 3x3 + ReLU + 2x2 maxpool

## Results (mean ± std across seeds)

| Arm | final test acc | max test acc | filters end | head synapses | conv grow/prune | verdict vs base |
|---|---|---|---|---|---|---|
| fixed-hand-k6 | 0.425 ± 0.028 | 0.453 ± 0.017 | 6.0 | 943 | 0.0/0.0 | (baseline) |
| learned-k6 | 0.453 ± 0.051 | 0.507 ± 0.012 | 6.0 | 911 | 0.0/0.0 | ~ |
| selfsize-2to12-rand | 0.414 ± 0.027 | 0.522 ± 0.029 | 12.0 | 1018 | 10.0/0.0 | ~ |

Verdict = 95% seed-bootstrap CI of the final-test-acc difference vs the baseline (UP/DOWN/~).

![acc](acc_curves.png)

![filter count](count_curves.png)

### fixed-hand-k6 learned filters

![fixed-hand-k6](filters_fixed-hand-k6.png)

### learned-k6 learned filters

![learned-k6](filters_learned-k6.png)

### selfsize-2to12-rand learned filters

![selfsize-2to12-rand](filters_selfsize-2to12-rand.png)
