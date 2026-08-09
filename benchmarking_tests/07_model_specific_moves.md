# Model-specific moves

## Question

Can exact TFIM symmetries or classical cluster updates reduce autocorrelation
without compromising detailed balance?

## Hypothesis

An exact global spin inversion can eliminate trapping between the two
longitudinal-magnetization sectors of the standard TFIM, although it cannot
directly reduce autocorrelation of spin-inversion-even observables such as
energy.

## Code/Environment

- Implementation: `mainqmc.hpp`, gated by `TFIM_GLOBAL_Z2_MOVE`.
- Planner control: `move=none|global_z2` and
  `configs/model_move_controls.json`.
- Ensemble summary: `results/model_move_representation_controls.json`;
  generator `experiments/summarize_model_controls.py`.
- Machine: Intel i5-6600K, four cores, WSL2 Linux, GCC/OpenMPI wrapper 9.4.
  The 12 retained stream traces total 276,615,221 bytes; per-run manifest and
  ordered trace-checksum digests are in the compact ensemble summary.
- Correctness protocol: `Z2-SMOKE-20260809`, summarized in
  `results/global_z2_smoke.json`.

## Protocol

For the standard basis TFIM,

```text
H = -sum_<ij> Zi Zj - Gamma sum_i Xi,
```

let `F` complement every computational-basis spin. Every `Zi Zj` eigenvalue is
unchanged by `F`, and `F` commutes with every `Xi` permutation. Therefore all
intermediate diagonal energies, the operator product, divided difference, and
complete PMR weight are unchanged. The proposal `C -> F(C)` is an involution
with a symmetric proposal probability, so detailed balance holds term by term
with acceptance one.

The implementation spends a configurable fraction of update attempts on this
move, flips both the live and base states without rebuilding weights, counts
accepted moves in traces/checkpoints, and is rejected by the planner for the
rotated or parity-restricted representation. The control matrix compares the
move on/off under matched seeds for fixed PMR and beta tempering.

Production protocols now enable the built-in longitudinal Z magnetization and
report its IAT and ESS separately. It is intentionally not folded into the
energy/H1/q/sign convergence maximum because it tests a different symmetry
mode and is odd under the move.

## Results

With the move enabled at probability 0.1, the two-spin fixed run executed 211
global inversions and the four-rank PT run executed 854. Fixed and all four PT
temperature marginals passed the preconfigured exact-diagonalization smoke
tolerance. All swap edges remained active (acceptance 0.754, 0.646, 0.440) and
126 round trips were observed. These data validate correctness and accounting,
not an autocorrelation improvement.

A matched four-seed control was then run for the 4x4 PBC TFIM at
`(Gamma,beta)=(1,4)`, using 10,000 warmup and 200,000 production updates per
chain, measurement interval 10, and cheap observables. Standard/no-move run IDs
were `ede67829c6691b4c`, `d18d44a3e618ced1`, `4567c5c7916d42c4`, and
`a019c4faab739fb3`; global-move IDs were `e2f14b2d931c3794`,
`7faa6f280e46096d`, `de684f46e11a933b`, and `138ccd6b7b7f0bab`.

The move reduced median longitudinal-magnetization IAT by 4.34x (matched-seed
bootstrap 95% interval 4.02--4.78) and raised its ESS/core-hour by 4.65x
(4.27--4.93). Median energy ESS/core-hour instead changed by a factor 0.94
(0.79--1.01). All matched energy means pass the five-combined-SE correctness
check, but only one of four move runs passes the full drift gate; therefore the
configured speedup claim correctly remains false.

The same seeds compared standard `ZZ+X` with historical rotated/parity `XX+Z`.
The observed medians favor standard by 42.0x in energy IAT, 12.1x in wall time,
and 516x in energy ESS/core-hour. None of the short rotated runs converged, and
their paired energy agreement fails, so this large practical ratio is also not
a formal speedup claim. Rotated run IDs were `a85af5d0bcc8bf11`,
`a667a29597691f7a`, `af68e84189b7203b`, and `05ecbaf30166d0fd`.

These runs were planned from base commit `d2848a1` while the magnetization
instrumentation in this vertical slice was uncommitted, so manifests correctly
record `dirty_worktree=true`. The exact generated parameters, staged C++
sources, commands, and checksums remain in each archive; the commit containing
this note records the final source state. Future empirical plans should either
require a clean tree or record a content hash of all staged source files rather
than relying on the base commit alone.

## Interpretation

The move is essentially free and exactly correct for the gated representation,
but energy, heat capacity, ES, and FS are even under global inversion. It should
not be advertised as a solution to their critical slowing; its primary direct
benefit is for symmetry-odd observables and sector exploration.

Replacing ordinary updates with symmetry flips is the wrong default for
energy-focused production: it produces a clean magnetization benefit but no
energy-efficiency gain and worsened the drift pass rate in this short pilot. A
free random inversion after an ordinary update should preserve the
magnetization benefit without sacrificing energy-changing attempts.

Representation choice is a much larger lever than this move at the historical
4x4 point. The standard representation lowers mean q from roughly 115 to 16
and turns a nonconverged short run into nearly exact energy estimates. The full
comparison still needs a converged rotated baseline before any numerical
speedup ratio is report-ready.

## Limitations

- The smoke test is only a two-spin, high-uncertainty correctness check.
- Three of four global-move runs fail the strict early/late drift test despite
  R-hat at most 1.0065; longer warmup or post-update implementation is needed.
- All four short rotated controls fail convergence, so their observed
  ESS/core-hour ratios are diagnostic rather than speedup claims.
- Three rotated controls were accidentally started concurrently, immediately
  stopped, and quarantined under
  `artifacts/model_move_controls_v2/invalid_concurrent_20260809`; only serial
  reruns are present in the summary and evidence.
- A classical Wolff move at Gamma=0 still needs its own implementation and
  validation.
- No general q>0 space-time cluster move is enabled; that remains gated behind
  a separate detailed-balance derivation and transition-matrix tests.

## Report-ready Claims

For the standard unrestricted TFIM, full computational-basis spin inversion is
an acceptance-one PMR move because it leaves every path weight unchanged.

At 4x4, Gamma 1, beta 4, the current replace-an-update implementation has a
bootstrap-resolved benefit for longitudinal-magnetization autocorrelation but
does not pass the preregistered production convergence gate or improve energy
ESS/core-hour. No empirical move speedup claim is made.

## Open Questions

- Should it be applied as a free post-update symmetry randomization instead of
  competing with ordinary PMR updates?
