# Frozen-model physical analysis protocol

Written before generating and inspecting the new replication dataset.

## Fixed choices

- Keep the trained four-dimensional PCA and seed-47 autoencoder from experiment v1.
- Do not tune their architecture, weights, checkpoint or data preprocessing.
- Generate new Wolff data at the same 11 temperatures with seed family 20271001–20271011,
  distinct from the original data family. Eight chains per temperature, 160 snapshots,
  burn=1200 and thinning=32 lazy cluster attempts. This tests new sampling streams at
  the same system and size, not different lattice sizes or universality classes.
- Report both the existing held-out sets and the new dataset, including per-temperature
  errors so temperature variation cannot hide poor within-temperature performance.

## Observables, computed after representation training

- Signed magnetization: mean spin.
- Absolute magnetization: absolute value of mean spin.
- C1: mean spin product at separation one along the two lattice axes. Energy/spin
  equals -2 C1 for this Hamiltonian, so energy is not counted as a separate discovery.
- C4: analogous mean product at separation four. This is a raw two-point correlation,
  not the connected correlation with the magnetization contribution subtracted.

Use a fixed degree-two polynomial readout of the four latent coordinates, followed
by feature standardization and Ridge(alpha=0.001), fit on the original training
chains only. Apply exactly the same readout family to PCA and AE. Labels are used
only for this post-hoc accessibility diagnostic, not for representation learning.
Report R² and RMSE globally and within each temperature. This is a restricted probe,
not a bound on all information that any nonlinear readout could recover.

## Interpretation checks

- Examine PCA loadings and their overlap with the normalized uniform-spin direction.
- Compare reconstruction error and observable readout on independent data.
- Give temperature-stratified chain-bootstrap intervals for paired AE minus PCA
  reconstruction error, conditional on the frozen models.
- Keep negative or inconclusive results. Do not adjust settings to make AE win.
- No critical point, critical exponent, phase-classifier accuracy or symmetry-discovery
  claim follows from these diagnostics alone.
