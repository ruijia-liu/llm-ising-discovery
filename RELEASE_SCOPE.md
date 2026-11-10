# Release scope

This source release contains the generalized-Ising laboratory, numerical teacher, teaching-message formats, student-training and evaluation implementations, protocols and checks. The earlier equilibrium/PCA/autoencoder project is a separate study and is not included.

## Included and excluded artifacts

- Included: Python source, experiment protocols, dependency specifications, and small example-generating commands.
- Generated locally: example messages, numerical trajectories, calibration, curricula and evaluation output.
- Excluded: virtual environments, model weights, adapters, training checkpoints, cached tensors, API configuration helpers, account-specific API experiment scripts, personal application documents, internal progress logs and unfinished manuscripts.

The source scripts are preserved at their original paths. Protocol text referencing an omitted research artifact identifies it as a local artifact rather than linking to a nonexistent public file. Machine-specific interpreter instructions have been replaced with portable Python commands. These documentation edits mean this release is not byte-identical to the historical source snapshots; scripts requiring historical source hashes retain their guards.

## Models and dependencies

Third-party packages and models retain their own licenses. Model weights are not redistributed. Model download helpers retrieve upstream model cards and license files alongside weights. No new blanket software or dataset license is assigned by this release.

The minimal environment pins NumPy 2.3.5. The model reproduction lock describes the recorded Windows/Python 3.12 environment and excludes the separately installed CUDA PyTorch wheel. It should be reviewed for the target platform before use.

## Claims supported by this release

Readers can inspect the physics and student interfaces, generate a causal teaching example and run the self-contained checks. The formal teaching-effect comparison is incomplete. Historical development observations are exploratory, and full training/inference or formal heldout reproduction is not claimed by the CPU example or tests.
