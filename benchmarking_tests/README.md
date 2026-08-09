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
| H01 | Historical reproduction | planned | pending | pending | — |
| E02 | Fast FS estimator | planned | pending | pending | — |
| A03 | Autocorrelation automation | smoke validated | this commit | `STREAM-SMOKE-20260809` | Independent signed stream traces and all-slot QCPT analysis validated |
| Q04 | QCPT schedule search | planned | pending | pending | — |
| T05 | Critical TFIM scaling | planned | pending | pending | — |
| S06 | SSE comparison | planned | pending | pending | — |
| M07 | Model-specific moves | correctness smoke passed | this commit | `Z2-SMOKE-20260809` | Acceptance-one global Z2 move gated to standard TFIM |

## Report assembly

`report_manifest.json` fixes the section order.  Run

```text
python3 benchmarking_tests/compile_report.py
```

to validate references and produce a combined Markdown source.  If Pandoc is
installed, add `--latex` to also build the LaTeX source.  Generated build files
are intentionally ignored; the notes, manifest, bibliography, and template are
the canonical sources.
