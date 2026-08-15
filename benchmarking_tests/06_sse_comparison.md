# Directed-loop SSE comparison

## Question

For energy and specific heat, how does PMR's time to precision compare with a
matched directed-loop SSE calculation?

## Hypothesis

Directed-loop SSE will have a substantially higher practical ceiling for the
standard sign-free TFIM, but a controlled comparison requires identical
Hamiltonian normalization, boundary conditions, core counts, and precision
criteria.

## Code/Environment

- Input adapter: `experiments/sse_adapter.py`.
- Container recipe and lock: `containers/alps-sse/`.
- Official stable source: ALPS `v2.3.4`, commit
  `97914eba01fb8eae1b96d460b577cb62a8f7ba94`.
- Documentation: [ALPS directed-loop SSE](https://alps.comp-phys.org/documentation/methods/qmc/sse_old/)
  and [ALPS TFIM convention](https://alps.comp-phys.org/tutorials/ed/ed04/).

## Protocol

ALPS uses spin operators `S=sigma/2`. To match the PMR Hamiltonian

```text
-J sigma_z sigma_z - Gamma sigma_x,
```

without rescaling beta or energy, generate `local_S=1/2`, `Jxy=0`, `Jz=-4J`,
and `Gamma_ALPS=2 Gamma`. Use the official `square lattice` and
`open square lattice` identifiers for PBC and OBC, respectively. Match seeds,
four cores, lattice, beta, thermalization policy, and requested precision; only
energy density and specific heat enter the comparative claim.

The container recipe pins stable ALPS `v2.3.4` and the Ubuntu base manifest.
After a build, `build.sh` records the engine inspection and prints repository
digests for transfer into `lock.json`.

## Results

Two unit tests pass for the Pauli-to-spin normalization, beta-to-temperature
conversion, square dimensions, and exact PBC/OBC lattice identifiers. The
container image was not built because this desktop environment has no Docker,
Podman, or Apptainer executable; consequently no image digest or SSE physics
result is claimed.

## Interpretation

The normalization and reproducible build boundary are explicit, removing a
factor-of-two/four ambiguity that would otherwise invalidate the comparison.

## Limitations

- `lock.json` intentionally has a null image digest until a container-capable
  host performs and records the build.
- The ALPS HDF5 result parser and PMR/SSE exact-value cross-check remain to be
  implemented after the image exists.
- The comparator does not assess PMR-only arbitrary-observable estimators.

## Report-ready Claims

None; the comparator has not been built or run.

## Open Questions

- How much of the practical ceiling gap is attributable to non-local SSE
  updates rather than representation and estimator cost?
