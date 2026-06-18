# Conv-SPROUT Phase 2 — conv-sprout-actprune-motifs

- **Dataset:** motifs  |  **Seeds:** 5  |  **Steps:** 60000  |  **Baseline:** fixed-hand-k6
- **Head:** sparse phasic (w32-sparse economy), conv 3x3 + ReLU + 2x2 maxpool

## Results (mean ± std across seeds)

| Arm | final test acc | max test acc | filters end | head synapses | conv grow/prune | verdict vs base |
|---|---|---|---|---|---|---|
| fixed-hand-k6 | 0.425 ± 0.028 | 0.453 ± 0.017 | 6.0 | 943 | 0.0/0.0 | (baseline) |
| selfsize-12to12-cos | 0.743 ± 0.026 | 0.751 ± 0.019 | 12.0 | 1284 | 0.0/0.0 | UP |
| selfsize-12to12-actprune-cos | 0.736 ± 0.034 | 0.748 ± 0.015 | 11.8 | 1200 | 0.0/0.0 | UP |

Verdict = 95% seed-bootstrap CI of the final-test-acc difference vs the baseline (UP/DOWN/~).

![acc](acc_curves.png)

![filter count](count_curves.png)

### fixed-hand-k6 learned filters

![fixed-hand-k6](filters_fixed-hand-k6.png)

### selfsize-12to12-cos learned filters

![selfsize-12to12-cos](filters_selfsize-12to12-cos.png)

### selfsize-12to12-actprune-cos learned filters

![selfsize-12to12-actprune-cos](filters_selfsize-12to12-actprune-cos.png)
