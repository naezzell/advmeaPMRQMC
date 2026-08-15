# Benchmarking evidence index

This directory is the paper trail for the `benchmark-speedup` branch.  The
machine-readable campaign output is produced by `experiments/study.py`; these
notes record what was tested, what was observed, and which statements are safe
to carry into the final report.

## Evidence rules

- Every empirical statement cites a Git commit and one or more content-derived
  run IDs.
- A run ID resolves to a manifest containing the code revision, Hamiltonian,
  configuration, command, seeds, software versions, and hardware information.
- Large raw traces are not committed.  Notes instead record their archive path,
  byte size, and SHA-256 digest.  Compact summaries and plots may be committed
  under `benchmarking_tests/results/`.
- `Observed result` is kept separate from `Interpretation` and from a
  `Report-ready claim`.  Inconclusive and negative results remain in the index.
- A speedup is report-ready only after a correctness gate and a seed-ensemble
  confidence interval that excludes a ratio of one.

## Experiment index

| ID | Topic | Status | Commit | Run IDs | Headline result |
|---|---|---|---|---|---|
| M00 | Methodology and provenance | ready | `b44818d` | n/a | Protocol and evidence schema defined |
| H01 | Historical reproduction | 3x3 and 8x8 sweeps reproduced; 4x4 cliff measured | this commit | IDs in note | All four 8x8 beta=1 Gamma points match archive; advanced precision remains limiting |
| E02 | Fast FS estimator | L=2 speedup established | this commit | 8 paired run IDs in note | 4.69x measurement and 3.87x wall median speedup |
| A03 | Autocorrelation automation | diagnostics corrected and cached | this commit | smoke IDs in note; cache uses 12 control runs | Cached repeat 1.06 s vs 263.04 s uncached; summary byte-identical |
| Q04 | QCPT schedule search | L=2 ranking and production complete | this commit | 24 retained IDs in note/result | Pure-beta wins grid utility; no resolved endpoint advantage over diagonals |
| T05 | Critical TFIM scaling | planned | pending | pending | — |
| S06 | SSE comparison | adapter ready, image build blocked | this commit | pending | ALPS normalization and source/base pins tested |
| M07 | Model-specific moves | four-seed pilot measured | this commit | 12 matched IDs in note | Z-magnetization ESS/core-hour 4.65x, but production gate fails; standard basis strongly favored |

## Report assembly

`report_manifest.json` fixes the section order.  Run

```text
python3 benchmarking_tests/compile_report.py
```

to validate references and produce a combined Markdown source.  If Pandoc is
installed, add `--latex` to also build the LaTeX source.  Generated build files
are intentionally ignored; the notes, manifest, bibliography, and template are
the canonical sources.
