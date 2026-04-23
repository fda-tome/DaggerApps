# SC26 paper — integrated branch `SC26_AD_AE`

This branch merges the per–case-study pins from the SC26 Artifact Description
(`paper/sc26_ad_submission.tex` in the paper repo) into a **single** line of
development for reviewers.

## Integration tip

The authoritative tip SHA is cited in the AD documents
(`paper/sc26_ad_submission.tex`, `paper/sc26_ad_appendix.md`,
`paper/sc26_artifact_description.md`) — reviewers should `git checkout`
that SHA rather than a branch head. Clone the branch with:

```bash
git clone --branch SC26_AD_AE https://github.com/fda-tome/DaggerApps.git
```

then `git checkout <sha-from-AD>` before `Pkg.instantiate()`.

## Source pins merged (short SHA → full SHA)

| Case study | Source branch | Short | Full commit |
|------------|---------------|-------|-------------|
| Stencil (Metal) | `main` | `793936d` | `793936d9053fcb7629cd71a187bafba2ec6a1c44` |
| LLM pass@k | `pass-at-k-study/remove-parsec-and-runtime-env-fixes` | `2fbcdcc` | `2fbcdcc59fdf75f160c9b696165563c2c04a59d4` |
| Barnes–Hut | `WIP-barnes-hut` | `134e1b7` | `134e1b78796f7a43fa75c006248b0d2a24fdd492` |
| Seam carving | `WIP-seam-carving` | `7564976` | `75649763d42bf86914457b7d2d4ee4411f540bea` |
| Cholesky (NVIDIA+Intel) | `WIP-cholesky` | `e182c31` | `e182c312e4578c4f708edb6eea85117725f80a50` |
| Cholesky (AMD) | `WIP-Cholesky-AMD` | `8cbef32` | `8cbef321016766d6975973b3af12787d1aba46de` |

**Merge order on `SC26_AD_AE`:** base = `main@793936d`, then `2fbcdcc`, `134e1b7`,
`7564976`, `e182c31`, `8cbef32` (each as a `git merge --no-ff <full-sha>`).

## One-line entrypoints (see AD appendix for parameters)

- **Cholesky:** `benchmarks/scripts/gpu-cholesky.jl`, `gpu-cholesky-sweep.jl`; PBS `scripts/polaris_cholesky_bench.pbs`; AMD `scripts/mi300a_cholesky_bench.slurm`
- **Barnes–Hut:** `benchmarks/scripts/barnes-hut.jl`
- **Seam carving:** `benchmarks/scripts/seam-westrick-scaling.jl`, driver `benchmarks/scripts/run-seam-scalability.sh`
- **Stencil:** `benchmarks/scripts/game-of-life.jl` or `heat-propagation` (see repo README)
- **Pass@k:** `apps/pass-at-k-study/scripts/run_full_study.sh`
