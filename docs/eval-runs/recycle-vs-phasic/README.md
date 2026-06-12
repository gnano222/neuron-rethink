# Evaluation run: recycle-vs-phasic

- **Date:** 2026-06-11 22:45:10
- **Variants:** currency, phasic, phasic-recycle  (baseline: phasic)
- **Seeds:** 5  |  **Dataset:** spirals  |  **Steps:** 15000 (+3000 shift)
- **Commit:** 2f5152f
- **Command:** `python evaluate.py --variants currency,phasic,phasic-recycle --seeds 5 --dataset spirals --steps 15000 --shift 3000 --baseline phasic --jobs 6 --no-cache --publish --run-name recycle-vs-phasic`

## Key metrics

| Metric | What it means | currency | phasic (baseline) | phasic-recycle |
|---|---|---|---|---|
| final_test_acc ↑ | held-out accuracy at the end of the run | 0.973 ± 0.014 ≈ | 0.963 ± 0.031 | 0.977 ± 0.021 ≈ |
| steps_to_90 ↓ | steps to first reach 90% test accuracy | 1801 ± 606.630 ≈ | 1721 ± 587.878 | 1721 ± 587.878 ≈ |
| steps_to_95 ↓ | steps to first reach 95% test accuracy | 2481 ± 785.875 ≈ | 2681 ± 881.816 | 2681 ± 881.816 ≈ |
| auc_test_acc ↑ | area under the test-accuracy curve (speed + level) | 0.934 ± 0.021 ≈ | 0.928 ± 0.026 | 0.926 ± 0.022 ≈ |
| pre_shift_test_acc ↑ | test accuracy just before the concept shift | 0.989 ± 0.010 ≈ | 0.981 ± 0.027 | 0.986 ± 0.012 ≈ |
| recovered_test_acc ↑ | test accuracy at the end, after the label swap | 0.973 ± 0.014 ≈ | 0.963 ± 0.031 | 0.977 ± 0.021 ≈ |
| synapse_count_end | live synapses at the end | 234.400 ± 9.394 ≈ | 123 ± 18.078 | 122.200 ± 20.605 ≈ |
| effective_density | live edges as a fraction of fully-connected | 0.407 ± 0.016 ≈ | 0.214 ± 0.031 | 0.212 ± 0.036 ≈ |
| ghost_dense_cost | candidate ghost wires the grow-scan must consider (~N²) | 729.600 ± 9.394 ≈ | 841 ± 18.078 | 841.800 ± 20.605 ≈ |
| ghost_pairs_scored | candidate wires actually scored after activity+demand pruning | 85.379 ± 14.711 ≈ | 81.664 ± 15.473 | 146.358 ± 42.332 ≈ |
| mean_neuron_activation | avg hidden-neuron ReLU output on test data (neuron value) | 0.221 ± 0.031 ≈ | 0.254 ± 0.046 | 0.323 ± 0.060 ≈ |
| dead_unit_frac ↓ | fraction of hidden neurons that never fire (scale-free) | 0.092 ± 0.028 ▲ | 0.179 ± 0.049 | 0.013 ± 0.017 ▲ |
| idle_unit_frac ↓ | fraction of hidden neurons dead OR outputless (not in service) | 0.125 ± 0.013 ▲ | 0.300 ± 0.043 | 0.321 ± 0.082 ≈ |
| n_recycle_events | dead-unit recycles fired over the run (sleep recycling) | 0 ± 0 ≈ | 0 ± 0 | 10.200 ± 2.315 ≈ |
| recycled_rehired_frac | of recycled units, fraction back in service at the end | — ± — ? | — ± — | 0.200 ± 0.400 ? |
| max_grows_into_one_neuron ↓ | most times one neuron was grown into (churn) | 17.800 ± 4.956 ≈ | 13.200 ± 2.786 | 12.800 ± 6.911 ≈ |
| oscillation_frac ↓ | fraction of grown edges grown ≥2× (thrash) | 0.139 ± 0.020 ≈ | 0.063 ± 0.126 | 0.015 ± 0.022 ≈ |
| freeloader_frac ↓ | fraction of synapses below the prune-utility floor | 0.052 ± 0.035 ≈ | 0.024 ± 0.012 | 0.057 ± 0.055 ≈ |
| conf_utility_corr ↑ | corr of confidence with real utility (calibration) | 0.064 ± 0.034 ≈ | 0.099 ± 0.113 | 0.213 ± 0.117 ≈ |
| dead_unit_count ↓ | hidden neurons that never fire on test data | 4.400 ± 1.356 ▲ | 8.600 ± 2.332 | 0.600 ± 0.800 ▲ |

## Full scorecard

| Metric | currency | phasic (baseline) | phasic-recycle |
|---|---|---|---|
| **Prediction performance** | | | |
| final_test_acc ↑ | 0.973 ± 0.014 ≈ | 0.963 ± 0.031 | 0.977 ± 0.021 ≈ |
| max_test_acc ↑ | 0.997 ± 0.003 ≈ | 0.998 ± 0.002 | 0.998 ± 0.003 ≈ |
| final_train_acc ↑ | 0.976 ± 0.018 ≈ | 0.966 ± 0.033 | 0.976 ± 0.023 ≈ |
| final_test_loss ↓ | 0.069 ± 0.041 ≈ | 0.117 ± 0.062 | 0.099 ± 0.057 ≈ |
| **Training efficacy** | | | |
| steps_to_90 ↓ | 1801 ± 606.630 ≈ | 1721 ± 587.878 | 1721 ± 587.878 ≈ |
| steps_to_95 ↓ | 2481 ± 785.875 ≈ | 2681 ± 881.816 | 2681 ± 881.816 ≈ |
| auc_test_acc ↑ | 0.934 ± 0.021 ≈ | 0.928 ± 0.026 | 0.926 ± 0.022 ≈ |
| final_acc_stability ↓ | 0.035 ± 0.017 ▲ | 0.062 ± 0.022 | 0.057 ± 0.023 ≈ |
| pre_shift_test_acc ↑ | 0.989 ± 0.010 ≈ | 0.981 ± 0.027 | 0.986 ± 0.012 ≈ |
| recovered_test_acc ↑ | 0.973 ± 0.014 ≈ | 0.963 ± 0.031 | 0.977 ± 0.021 ≈ |
| recovery_gap ↓ | 0.016 ± 0.014 ≈ | 0.018 ± 0.049 | 0.009 ± 0.026 ≈ |
| recovery_steps ↓ | ∞ ± — ? | ∞ ± — | ∞ ± — ? |
| **Synapse structure** | | | |
| synapse_count_start | 244 ± 0.894 ≈ | 242 ± 0.894 | 242 ± 0.894 ≈ |
| synapse_count_peak | 247.800 ± 4.167 ≈ | 242 ± 0.894 | 242 ± 0.894 ≈ |
| synapse_count_end | 234.400 ± 9.394 ≈ | 123 ± 18.078 | 122.200 ± 20.605 ≈ |
| n_grow_events | 130.600 ± 10.911 ≈ | 59.600 ± 9.091 | 67.200 ± 13.257 ≈ |
| n_prune_events | 138.200 ± 7.909 ≈ | 178.600 ± 10.012 | 167 ± 5.477 ≈ |
| distinct_neurons_grown | 19.200 ± 2.786 ≈ | 13.600 ± 1.744 | 15.600 ± 1.855 ≈ |
| turnover ↓ | 1.115 ± 0.065 ≈ | 1.180 ± 0.053 | 1.164 ± 0.060 ≈ |
| max_grows_into_one_neuron ↓ | 17.800 ± 4.956 ≈ | 13.200 ± 2.786 | 12.800 ± 6.911 ≈ |
| mean_fan_in | 4.688 ± 0.188 ≈ | 2.460 ± 0.362 | 2.444 ± 0.412 ≈ |
| mean_fan_out | 4.688 ± 0.188 ≈ | 2.460 ± 0.362 | 2.444 ± 0.412 ≈ |
| effective_density | 0.407 ± 0.016 ≈ | 0.214 ± 0.031 | 0.212 ± 0.036 ≈ |
| **Synapse quality** | | | |
| p10_utility ↑ | 0.580 ± 0.050 ▼ | 0.767 ± 0.068 | 0.740 ± 0.184 ≈ |
| freeloader_frac ↓ | 0.052 ± 0.035 ≈ | 0.024 ± 0.012 | 0.057 ± 0.055 ≈ |
| mean_survivor_age ↑ | 15590 ± 226.611 ≈ | 14259 ± 1902 | 12891 ± 2054 ≈ |
| median_survivor_age ↑ | 18000 ± 0 ≈ | 18000 ± 0 | 14960 ± 6080 ≈ |
| mean_pruned_lifespan | 4976 ± 147.466 ≈ | 10985 ± 1654 | 11007 ± 1721 ≈ |
| oscillation_frac ↓ | 0.139 ± 0.020 ≈ | 0.063 ± 0.126 | 0.015 ± 0.022 ≈ |
| max_regrow ↓ | 3.400 ± 0.490 ▼ | 0.400 ± 0.800 | 0.400 ± 0.490 ≈ |
| conf_utility_corr ↑ | 0.064 ± 0.034 ≈ | 0.099 ± 0.113 | 0.213 ± 0.117 ≈ |
| frozen_freeloader_frac ↓ | 0 ± 0 ≈ | 0 ± 0 | 0 ± 0 ≈ |
| dead_unit_count ↓ | 4.400 ± 1.356 ▲ | 8.600 ± 2.332 | 0.600 ± 0.800 ▲ |
| dead_unit_frac ↓ | 0.092 ± 0.028 ▲ | 0.179 ± 0.049 | 0.013 ± 0.017 ▲ |
| idle_unit_frac ↓ | 0.125 ± 0.013 ▲ | 0.300 ± 0.043 | 0.321 ± 0.082 ≈ |
| mean_neuron_activation | 0.221 ± 0.031 ≈ | 0.254 ± 0.046 | 0.323 ± 0.060 ≈ |
| inert_synapse_frac ↓ | 0 ± 0 ≈ | 0 ± 0 | 0 ± 0 ≈ |
| used_vs_allocated | 0.969 ± 0.037 ≈ | 0.508 ± 0.076 | 0.505 ± 0.086 ≈ |
| n_recycle_events | 0 ± 0 ≈ | 0 ± 0 | 10.200 ± 2.315 ≈ |
| recycled_rehired_frac | — ± — ? | — ± — | 0.200 ± 0.400 ? |
| **Compute cost** | | | |
| ghost_dense_cost | 729.600 ± 9.394 ≈ | 841 ± 18.078 | 841.800 ± 20.605 ≈ |
| ghost_pairs_scored | 85.379 ± 14.711 ≈ | 81.664 ± 15.473 | 146.358 ± 42.332 ≈ |
| **Signal sanity** | | | |
| meter_fidelity ↑ | 0.894 ± 0.083 ≈ | 0.931 ± 0.056 | 0.916 ± 0.090 ≈ |

Baseline: **phasic**. ▲ better / ▼ worse / ≈ no clear difference vs baseline (95% bootstrap CI of the mean difference). Cells show mean ± std across seeds.

## Charts

### acc_curves
![acc_curves](acc_curves.png)

### churn_curves
![churn_curves](churn_curves.png)

### cost_scaling
![cost_scaling](cost_scaling.png)

### count_curves
![count_curves](count_curves.png)

### quality_currency
![quality_currency](quality_currency.png)

### quality_phasic-recycle
![quality_phasic-recycle](quality_phasic-recycle.png)

### quality_phasic
![quality_phasic](quality_phasic.png)

### verdict_heatmap
![verdict_heatmap](verdict_heatmap.png)

