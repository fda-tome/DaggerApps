# Data

Store input datasets and generated artifacts here **per app**.

- Put assets for app `<app>` under `benchmarks/data/<app>/` (where `<app>` matches the folder name under `apps/`).
- Document origin, licensing, and any preprocessing steps.
- Prefer linking to large/immutable datasets instead of committing them to git.

Note: current seam‑carving, game-of-life, and heat-propagation benchmarks use synthetic inputs by default, so this folder is optional unless you customize a benchmark to read real datasets.
