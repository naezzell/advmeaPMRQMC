# Beta-Annealed Initialization Implementation Report

## Outcome

The fixed, beta-only parallel-tempering, and QCPT runners now support opt-in
beta-annealed initialization. Existing no-anneal execution remains on its
original code path: `Tsteps` retains its static-equilibration meaning, PT swap
timing and RNG consumption are unchanged, and legacy checkpoint bytes remain
unchanged.

When annealing is enabled, `Tsteps` is the ramp. Measurements are suppressed
during it. PT/QCPT swaps and flow accounting are also suppressed, and
production measurements and exchanges start immediately at the endpoint.
Each PT/QCPT slot has an independent beta curve; tau preserves its final
`tau/beta` ratio, while QCPT gamma remains fixed at the slot target.

## User interfaces

All three runner families accept:

- `--beta-anneal`
- `--anneal-start-factor F` (default `0.001`)
- `--anneal-interval K` (default `N` update calls)
- `--beta-anneal-schedule FILE`

Fixed runners additionally accept `--target-beta B` and `--target-tau T`;
the latter defaults to `B/2`. Absolute schedules use cumulative update
coordinates followed by one beta column for fixed runs or one column per
tempering slot.

`experiments/beta_anneal_driver.py` compiles a fixed runner once and launches
a fresh process for every requested target. It isolates target outputs and
writes a manifest containing runtime targets, ramp configuration, SHA-256
plan/schedule identities, observed seeds, commands, and wall times. Repeated
`--absolute-schedule BETA:FILE` arguments select per-target schedules.

## Design and compatibility

- `beta_anneal.hpp` owns parsing, validation, interpolation, retarget
  boundaries, and deterministic hashing.
- `retarget_run_parameters` changes beta/tau tables and rebuilds the current
  PMR weight without reinitializing the configuration or consuming RNG state.
- Tempering checkpoints incorporate the annealing plan into schedule identity.
  Mid-ramp resume restores the snapshot and reconstructs its beta/tau from the
  saved update coordinate.
- Dynamic fixed runs append identified runtime-parameter metadata to their
  legacy checkpoint payload. Static no-anneal runs do not append it, preserving
  their existing checkpoint bytes.
- Production-relative PT exchange parity is reset at the ramp endpoint. No
  warm-up exchange attempts or exchange RNG draws occur.

## Validation performed

- C++ component tests cover automatic ramps, interpolation, interval
  boundaries, tau scaling assumptions, endpoint forcing, hashing, mutual
  exclusion, insufficient plateaus, and malformed schedule coordinates,
  columns, values, monotonicity, and endpoints.
- The PMR retarget probe confirms the rebuilt current weight equals a direct
  target-parameter weight from the same snapshot and that target-weight
  evaluation itself is state/RNG preserving.
- The deterministic two-spin annealed validation passed the fixed endpoint and
  all four beta-only PT ladder points against exact diagonalization. The first
  production trace used update `Tsteps + stepsPerMeasurement`; swap counts
  contained no warm-up attempts.
- Interrupted/resumed mid-ramp beta PT and QCPT runs matched uninterrupted
  observables, swaps, and flow output exactly.
- An interrupted/resumed fixed run produced a byte-identical completed
  checkpoint to its uninterrupted control after excluding only the elapsed
  time field; this comparison includes observables, PMR state, and RNG state.
- The intended-size annealed QCPT validation passed fixed and QCPT estimates
  for every `(beta,gamma)` path point against exact diagonalization.
- The QCPT edge-case suite, including pure-beta/pure-gamma limits, split
  Hamiltonian layouts, repeated energies, cancellations, and retarget weight
  checks, passed.
- Existing no-anneal two-spin validation passed without changing its CLI or
  expected execution path.

## Commit structure

- `85377bb` — shared beta-annealing schedule primitives and unit coverage
- `ebc0648` — fixed/PT/QCPT runner and checkpoint integration
- `10d9ef1` — fixed-target driver and annealing validation workflows
- `a641690` — per-target `BETA:FILE` schedule mapping correction
- `858cefc` — exhaustive malformed-schedule tests
- `89ef0e0` — fixed-run mid-anneal checkpoint/resume validation

The untracked `benchmarking_tests/` tree was intentionally left untouched.
