# Critical square-TFIM scaling

## Question

How do the practical energy, specific-heat, ES, and FS ceilings scale near the
critical point when beta is scaled with `L`?

## Hypothesis

The standard `ZZ+X` representation should be the default scaling basis because
its expansion order and mixing cost are substantially lower than the
historical rotated/parity basis at low temperature. This must be confirmed
across size and critical Gamma rather than inferred from one 4x4 point.

## Code/Environment

Representation pilot details and run IDs are recorded in
`07_model_specific_moves.md` and
`results/model_move_representation_controls.json`.

## Protocol

Pending implementation.

## Results

A four-seed 4x4 pilot at Gamma 1 and beta 4 observed median standard/rotated
ratios of 42x for energy IAT, 12.1x for wall time, and 516x for energy
ESS/core-hour. All short rotated runs failed convergence, so these are ceiling
diagnostics rather than qualified speedups.

## Interpretation

The standard representation is the justified default for the critical pilot.
The rotated/parity representation remains only a historical anchor/control.

## Limitations

- This point is away from the standard square-TFIM critical Gamma and has only
  L=4.
- The rotated baseline must converge before a formal representation-speedup
  claim is possible.

## Report-ready Claims

No scaling claim yet. The representation pilot determines the production
default but does not establish behavior with L.

## Open Questions

- Which cost component first prevents scaling beyond `L=8`?
