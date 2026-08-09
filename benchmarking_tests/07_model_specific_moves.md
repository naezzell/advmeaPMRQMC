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

## Results

With the move enabled at probability 0.1, the two-spin fixed run executed 211
global inversions and the four-rank PT run executed 854. Fixed and all four PT
temperature marginals passed the preconfigured exact-diagonalization smoke
tolerance. All swap edges remained active (acceptance 0.754, 0.646, 0.440) and
126 round trips were observed. These data validate correctness and accounting,
not an autocorrelation improvement.

## Interpretation

The move is essentially free and exactly correct for the gated representation,
but energy, heat capacity, ES, and FS are even under global inversion. It should
not be advertised as a solution to their critical slowing; its primary direct
benefit is for symmetry-odd observables and sector exploration.

## Limitations

- The smoke test is only a two-spin, high-uncertainty correctness check.
- The planned matched-seed autocorrelation comparison has not been run.
- A classical Wolff move at Gamma=0 still needs its own implementation and
  validation.
- No general q>0 space-time cluster move is enabled; that remains gated behind
  a separate detailed-balance derivation and transition-matrix tests.

## Report-ready Claims

For the standard unrestricted TFIM, full computational-basis spin inversion is
an acceptance-one PMR move because it leaves every path weight unchanged.

## Open Questions

- Does the move reduce longitudinal-magnetization IAT enough to compensate for
  the update attempts it replaces?
- Should it be applied as a free post-update symmetry randomization instead of
  competing with ordinary PMR updates?
