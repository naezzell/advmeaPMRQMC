# Report outline

1. Motivation and historical context
2. Reproducibility and statistical methodology
3. Correctness reproduction of prior PMR-QMC results
4. Complexity and end-to-end impact of the fast FS estimator
5. Automated warmup, autocorrelation, measurement, and beta selection
6. QCPT schedule search and equal-resource controls
7. Square-lattice TFIM scaling in `L`, `lambda`, and `beta`
8. Comparison with directed-loop SSE for shared observables
9. Model-specific moves and remaining bottlenecks
10. Conclusions and recommended desktop-to-cluster workflow

## Claim-to-evidence registry

| Claim ID | Proposed claim | Required evidence | Status |
|---|---|---|---|
| C-E02 | The fast estimator reduces FS measurement cost | Slow/fast equality plus timing ensemble over `q` | pending |
| C-Q04 | A QCPT schedule improves target sampling | Exact gate plus equal-core ESS/hour CI | pending |
| C-T05 | PMR reaches larger critical square TFIMs reliably | Beta-converged multi-seed results | pending |
| C-S06 | PMR's practical ceiling differs from SSE | Matched observable/time-to-precision comparison | pending |
| C-M07 | A model-specific move reduces autocorrelation | Detailed balance tests plus seed ensemble | pending |
