# DeepEvolve TIG Knapsack (Rust)

Workspace: `examples/deepevolve/tig_knapsack`

## Prerequisites
- Python 3.10+ with `pip`/`venv` (or Conda)  
- Rust toolchain with Cargo (`rustup` stable works; run `rustup component add clippy rustfmt` if missing)  
- `python3`, `cargo`, `git`, and build essentials on your PATH  
- OpenAI API key (`OPENAI_API_KEY`) for DeepEvolve  

`deepevolve_interface.py` auto-detects the TIG repo path by searching upward for `tig.py`. If you keep the repo elsewhere or run from a different checkout, set `export TIG_REPO_ROOT=/path/to/tig-evolve` before launching DeepEvolve.

DeepEvolve itself now lives outside this repository; clone it from https://github.com/Daniel-T-S-Adams/deepevolve.git and set `DEEPEVOLVE_HOME` to your clone directory (for example, `export DEEPEVOLVE_HOME="$HOME/src/deepevolve"`).

## How to run the TIG knapsack example

1) Initialize the TIG challenge (run once from repo root)  
   ```
   python3 tig.py init_challenge knapsack
   # Optional: reuse a published solver as a starting point
   # python3 tig.py download_algorithm knapsack <algorithm_name>
   ```
   This writes the challenge spec/instances to `algo-runner/src/challenge/` and installs the starter solver under `algo-runner/src/algorithm/`, which DeepEvolve will overwrite each iteration.

2) Environment + dependencies  

   These instructions assume DEEPEVOLVE_HOME already points to a local checkout; clone the repo and export the variable before continuing.

   ```
   export DEEPEVOLVE_HOME="${DEEPEVOLVE_HOME:-$HOME/src/deepevolve}"
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r "$DEEPEVOLVE_HOME/requirements-mini.txt"
   export OPENAI_API_KEY="your-api-key-here"
   ```
   (See `$DEEPEVOLVE_HOME/README.md` if you prefer Conda or the minimal requirements file.)

3) Initial code and metadata  
   - If you use the provided starter, keep the files under `initial_code/`.  
   - If you bring your own algorithm, place your Rust sources in `initial_code/algo-runner/src/algorithm/`. You must have an `info.json`; if `initial_idea.json` and `initial_metrics.json` are missing, DeepEvolve will generate and cache them on first run (you can supply your own if you prefer).

4) Config tweaks  
   Edit `examples/deepevolve/tig_knapsack/config_tig.yaml` to change models, iteration counts, checkpoints, etc.  
   The `query` field is defined directly in this config (using a YAML block string), so you can update the evolution prompt there instead of passing it through the CLI.

5) Track/tests/timeout  
   Set these in your shell before running (they are read by `deepevolve_interface.py`):  
   ```
   export TIG_TRACK_ID="n_items=500,density=5"
   export TIG_NUM_TESTS=10
   export TIG_TIMEOUT=60
   ```
   `tig.py test_algorithm` will use these each evaluation.

6) Run 

   from inside your venv :
   

   ```
  cd /root/evolve/tig-evolve

   python "$DEEPEVOLVE_HOME/deepevolve.py" \
     --config-path "$PWD/examples/deepevolve/tig_knapsack" \
    --config-name config_tig \
    query="$(< examples/deepevolve/tig_knapsack/query.md)" \
    problem="tig_knapsack" \
    workspace="examples/deepevolve" \
    checkpoint="ckpt_tig_knapsack"
   ```


Notes:
- Evaluation uses `python tig.py build_algorithm` then `python tig.py test_algorithm ...`.
- Artifacts/checkpoints are written under `examples/deepevolve/tig_knapsack/ckpt_tig_knapsack/`.
- The evolution query can be customized by editing `query.md` in this directory.
