# Validated environment

Release checks were executed on 2026-08-01 with the versions below. This table
describes only the environment used to validate the anonymous code archive.

| component | validated version |
|---|---:|
| Python | 3.9.25 |
| PyTorch | 2.5.1+cu124 |
| NumPy | 1.26.4 |
| pandas | 2.3.3 |
| PyYAML | 6.0.3 |
| Hydra Core | 1.2.0 |
| Transformers | 4.57.6 |
| pytest | 8.4.2 |

The public ReCogDrive/NAVSIM environment supplies nuPlan, benchmark data,
metric caches, and model/evaluator-specific dependencies. GPU driver and CUDA
runtime compatibility must be checked on the target system before inference.
