# tig-evolve

Evolve Rust algorithms for TIG challenges. The `algo-runner` crate hosts your Rust solver; `tig.py` handles fetching challenges/algorithms and building/testing. Genetic/evolution agents live under `examples/`.

## Quickstart
- Pick a challenge and download its starter kit (choices: `satisfiability`, `vehicle_routing`, `knapsack`):
  - `python3 tig.py init_challenge <challenge>`
  - This writes the challenge description to `algo-runner/src/challenge/README.md`, baseline reference code to `algo-runner/src/challenge/baselines/`, and a boilerplate solver to `algo-runner/src/algorithm/mod.rs`.
- Optional: inspect an existing solver for the challenge:
  - `python3 tig.py download_algorithm <challenge> <algorithm_name>` (use `--force` to overwrite)
  - The downloaded Rust code replaces `algo-runner/src/algorithm/mod.rs`.
- Choose an evolution framework from `examples/` and follow its README. Each example expects the challenge to be initialized first and will iterate on the Rust sources in `algo-runner/src/algorithm/`.
- Build/test locally when needed:
  - `python3 tig.py build_algorithm` to compile `algo-runner` (requires Rust/Cargo).
  - `python3 tig.py test_algorithm <track_id> --tests N --workers W --timeout 60` to run solver evaluations; see example READMEs for recommended arguments and environment overrides.

## Additional commands
- `python3 tig.py list_algorithms <challenge>` lists available published solvers.
- `python3 tig.py download_algorithm ...` can be rerun with `--force` to reset your working solver.

See `algo-runner/src/challenge/README.md` for the active challenge specification and each `examples/<framework>/*/README.md` for framework-specific instructions.