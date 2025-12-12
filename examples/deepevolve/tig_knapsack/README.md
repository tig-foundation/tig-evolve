# DeepEvolve TIG Knapsack (Rust)

Workspace: `examples/deepevolve/tig_knapsack`

Run with the tig config (`deepevolve/configs/config_tig.yaml`):
- Set your OpenAI key (e.g., `export OPENAI_API_KEY=...`).
- From repo root:
  ```
 python deepevolve/deepevolve.py \
  --config-name config_tig \
  query="'Evolve the Rust TIG knapsack solver to maximize quality and reduce runtime.'" \
  problem="tig_knapsack" \
  workspace="examples/deepevolve" \
  checkpoint="ckpt_tig_knapsack"
  ```

Notes:
- Evaluation uses `python tig.py build_algorithm` then `python tig.py test_algorithm n_items=500,density=5` with `--tests 1 --timeout 60`.
- Override track/tests/timeout via env: `TIG_TRACK_ID`, `TIG_NUM_TESTS`, `TIG_TIMEOUT`.
- Evolved Rust sources are copied into `algo-runner/src/algorithm/` each iteration. Ensure the TIG challenge is initialized for knapsack.***

