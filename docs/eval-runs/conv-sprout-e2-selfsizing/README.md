# Conv-SPROUT Phase 2 — conv-sprout-e2-selfsizing

- **Dataset:** mnist  |  **Seeds:** 5  |  **Steps:** 15000  |  **Baseline:** fixed-hand-k6
- **Head:** sparse phasic (w32-sparse economy), conv 3x3 + ReLU + 2x2 maxpool

## Results (mean ± std across seeds)

| Arm | final test acc | max test acc | filters end | head synapses | conv grow/prune | verdict vs base |
|---|---|---|---|---|---|---|
| fixed-hand-k6 | 0.897 ± 0.015 | 0.912 ± 0.008 | 6.0 | 2008 | 0.0/0.0 | (baseline) |
| learned-k6 | 0.894 ± 0.029 | 0.907 ± 0.018 | 6.0 | 2165 | 0.0/0.0 | ~ |
| selfsize-2to12-split | 0.872 ± 0.018 | 0.893 ± 0.014 | 9.2 | 1914 | 7.2/0.0 | DOWN |
| selfsize-2to12-rand | 0.881 ± 0.010 | 0.893 ± 0.016 | 8.4 | 1795 | 6.4/0.0 | ~ |
| selfsize-12to12-prune | 0.888 ± 0.013 | 0.907 ± 0.007 | 12.0 | 3856 | 0.0/0.0 | ~ |

Verdict = 95% seed-bootstrap CI of the final-test-acc difference vs the baseline (UP/DOWN/~).

![acc](acc_curves.png)

![filter count](count_curves.png)

### fixed-hand-k6 learned filters

![fixed-hand-k6](filters_fixed-hand-k6.png)

### learned-k6 learned filters

![learned-k6](filters_learned-k6.png)

### selfsize-2to12-split learned filters

![selfsize-2to12-split](filters_selfsize-2to12-split.png)

### selfsize-2to12-rand learned filters

![selfsize-2to12-rand](filters_selfsize-2to12-rand.png)

### selfsize-12to12-prune learned filters

![selfsize-12to12-prune](filters_selfsize-12to12-prune.png)
