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
| H01 | Historical reproduction | 3x3 reproduced; 4x4 cliff measured | this commit | 3x3 IDs in note; 4x4 `561575c5085e1fb4`, `60e440b3b6219d00`, `31aeee8c9776254f` | 4x4 beta 0.1→4 loses 26,246x ESS/core-hour; endpoints match archive |
| E02 | Fast FS estimator | L=2 speedup established | this commit | 8 paired run IDs in note | 4.69x measurement and 3.87x wall median speedup |
| A03 | Autocorrelation automation | discrete R-hat corrected | this commit | `aee1cea9a2fae828`, `9d3f3c6f3c092919`, `a40a5b4376d6c288`, `b163f97dd43e5b40` | Largest smoke R-hat is 1.001; blocking/precision still fail |
| Q04 | QCPT schedule search | sweep-aware selector implemented | this commit | pending | Winner can optimize declared figure-grid ESS, coverage, or endpoint ESS |
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
