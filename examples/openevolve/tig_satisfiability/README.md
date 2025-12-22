# TIG Knapsack Example with OpenEvolve

This example uses [OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve) to optimize a Rust-based quadratic knapsack algorithm using The Innovation Game (TIG) protocol for instance generation and evaluation.

## Project Structure

```
tig-evolve/
├── tig.py                      # TIG CLI - build, test, download algorithms
├── algo-runner/                # Rust project that executes the algorithm
│   ├── Cargo.toml
│   └── src/
│       ├── main.rs             # CLI runner for solving and evaluating
│       ├── challenge/          # Challenge definition (initialized by tig.py)
│       └── algorithm/
│           └── mod.rs          # Algorithm code (copied from evolved programs)
└── examples/openevolve/tig_knapsack/
    ├── run.sh                  # Setup and run script (clones OpenEvolve, installs deps)
    ├── initial_program.rs      # Starting algorithm (simple greedy by value/weight ratio)
    ├── evaluator.py            # OpenEvolve evaluator - builds, runs, and scores algorithms
    ├── config.yaml             # OpenEvolve configuration (LLM settings, prompts, evolution params)
    └── openevolve_output/      # Output directory (created after running)
        ├── best/               # Best performing algorithm
        ├── checkpoints/        # Periodic snapshots
        └── logs/               # Execution logs
```

## Quick Start


The `run.sh` script handles all setup automatically:

```bash
cd /root/tig-evolve

# Set your API key (Google Gemini is used by default in config.yaml)
export OPENAI_API_KEY="your-google-ai-studio-api-key"

# Run the script
bash examples/openevolve/tig_knapsack/run.sh
```

The script will:
1. Clone OpenEvolve (v0.2.23) if not present
2. Create a Python virtual environment
3. Install OpenEvolve and dependencies
4. Initialize the knapsack challenge
5. Run the evolution


## Testing the Evaluator

To test `evaluator.py` directly on the initial program:

```bash
cd /root/tig-evolve/examples/openevolve/tig_knapsack
python3 -c "
from evaluator import evaluate
result = evaluate('initial_program.rs')
print('Result:', result)
"
```

## Testing TIG Algorithms with tig.py

The `tig.py` script in the tig-evolve root provides utilities for working with TIG algorithms.

### List Available Algorithms

```bash
cd /root/tig-evolve
python3 tig.py list_algorithms knapsack
```

### Download an Algorithm

```bash
python3 tig.py download_algorithm knapsack <algorithm_name> --force
```

### Build the Algorithm

```bash
python3 tig.py build_algorithm
```

### Test the Algorithm

```bash
python3 tig.py test_algorithm "n_items=500,density=25" --tests 100 --workers 4
```

## Testing the Best OpenEvolve Program

After running OpenEvolve, test the best evolved program:

```bash
cd /root/tig-evolve

# Copy the best program to algo-runner
cp examples/openevolve/tig_knapsack/openevolve_output/best/best_program.rs algo-runner/src/algorithm/mod.rs

# Build
python3 tig.py build_algorithm

# Test
python3 tig.py test_algorithm "n_items=500,density=25" --tests 1000
```

Or use the evaluator directly:

```bash
cd /root/tig-evolve/examples/openevolve/tig_knapsack
python3 -c "
from evaluator import evaluate
result = evaluate('openevolve_output/best/best_program.rs')
print('Result:', result)
"
```

## Configuration

Edit `config.yaml` to customize:

- **LLM settings**: Model selection, API base URL, temperature
- **Evolution parameters**: Population size, islands, migration rates
- **Prompts**: System message with optimization guidance
- **Evaluation**: Timeout, parallel evaluations

The default config uses Google Gemini models. To use other providers, update `llm.api_base` and `llm.primary_model`/`llm.secondary_model`.

## Metrics

The evaluator returns:
- **avg_btb**: Average "better than baseline" - percentage improvement over baseline (PRIMARY)
- **combined_score**: Score (...)
- **eval_time**: Total execution time in seconds
- **memory**: Total memory usage in KB


## Viewing Prompts from Checkpoints

Inspect the prompts used to generate evolved programs:

```bash
# Find program JSON files
ls examples/openevolve/tig_knapsack/openevolve_output/checkpoints/<checkpoint_number>/programs/

# View a specific program's prompts (if view_prompt.py exists)
python3 examples/openevolve/tig_knapsack/view_prompt.py \
    examples/openevolve/tig_knapsack/openevolve_output/checkpoints/<checkpoint_number>/programs/<program_id>.json
```

**Note:** Prompts are saved if `include_prompts: true` is set in `config.yaml`. Generation 0 programs won't have prompts since they're the initial program.
