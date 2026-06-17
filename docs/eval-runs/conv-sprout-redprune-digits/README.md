# Conv-SPROUT Phase 2 — conv-sprout-redprune-digits

- **Dataset:** mnist  |  **Seeds:** 5  |  **Steps:** 60000  |  **Baseline:** fixed-hand-k6
- **Head:** sparse phasic (w32-sparse economy), conv 3x3 + ReLU + 2x2 maxpool

## Results (mean ± std across seeds)

| Arm | final test acc | max test acc | filters end | head synapses | conv grow/prune | verdict vs base |
|---|---|---|---|---|---|---|
| fixed-hand-k6 | 0.931 ± 0.012 | 0.939 ± 0.012 | 6.0 | 1243 | 0.0/0.0 | (baseline) |
| selfsize-12to12-cos | 0.955 ± 0.009 | 0.957 ± 0.009 | 12.0 | 2446 | 0.0/0.0 | UP |
| selfsize-12to12-redprune-cos | 0.956 ± 0.011 | 0.958 ± 0.008 | 11.4 | 2250 | 0.0/0.0 | UP |

Verdict = 95% seed-bootstrap CI of the final-test-acc difference vs the baseline (UP/DOWN/~).

![acc](acc_curves.png)

![filter count](count_curves.png)

### fixed-hand-k6 learned filters

![fixed-hand-k6](filters_fixed-hand-k6.png)

### selfsize-12to12-cos learned filters

![selfsize-12to12-cos](filters_selfsize-12to12-cos.png)

### selfsize-12to12-redprune-cos learned filters

![selfsize-12to12-redprune-cos](filters_selfsize-12to12-redprune-cos.png)
