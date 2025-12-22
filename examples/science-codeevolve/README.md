# Evolving Algorithm for TIG with CodeEvolve

This example uses [CodeEvolve](https://github.com/inter-co/science-codeevolve.git) to optimize algorithms for TIG.

## Supported TIG Challenges

| Challenge | Description |
|-----------|-------------|
| **knapsack** | Quadratic knapsack problem |
| **vehicle_routing** | Vehicle routing problem with time windows |
| **satisfiability** | Boolean satisfiability 3-SAT problem |

## Requirements

- Python 3.13+
- Rust (install from [rustup.rs](https://rustup.rs))
- API Key for OpenAI API provider (we recommend [Google AI Studio](https://aistudio.google.com) for Gemini)

## Quick Start

The `run.sh` script handles all setup automatically:

```bash
# Set your API key (example uses Google Gemini API by default)
export API_KEY="your-google-ai-studio-api-key"

# Run the script from tig-evolve root
bash examples/science-codeevolve/run.sh <challenge>
```

Replace `<challenge>` with one of the supported challenges (e.g., `knapsack`, `vehicle_routing`, `satisfiability`).

The script will:
1. Clone CodeEvolve
2. Create a Python virtual environment
3. Install dependencies
4. Initialize the specified challenge
5. Start the evolution process

### Testing the Best Program

After running CodeEvolve, the best program is saved in the `experiments/config/<run_id>/` folder. To test this:

```bash
# Copy the best program to algo-runner
cp examples/science-codeevolve/tig_<challenge>/experiments/config/<run_id>/best_sol.rs algo-runner/src/algorithm/mod.rs

python3 tig.py build_algorithm
python3 tig.py test_algorithm <track> --tests 1000
```

Replace `<challenge>` with your challenge name, `<run_id>` with the experiment run ID (e.g., `0`, `1`), and `<track>` with the appropriate track for your challenge (e.g., `"n_items=500,density=25"` for knapsack).

Alternatively, this can be done programmatically in Python:

```bash
cd examples/science-codeevolve/tig_<challenge>/input
python3 -c "
from evaluate import evaluate
evaluate('../experiments/config/<run_id>/best_sol.rs', 'results.json')
"
cat results.json
```

## Configuring CodeEvolve

Each challenge has its own `config.yaml` located at `examples/science-codeevolve/tig_<challenge>/configs/config.yaml`. Edit it to customize:

- **LLM settings**: Model selection, API base URL, temperature
- **Evolution parameters**: Population size, iterations, number of runs
- **Prompts**: System message with optimization guidance
- **Evaluation**: Timeout, parallel evaluations

The default config uses Google Gemini models. To use other providers, update `llm.api_base` and `llm.primary_model`/`llm.secondary_model`.

## Configuring Evaluation Metrics

Each challenge has its own `evaluate.py` located at `examples/science-codeevolve/tig_<challenge>/input/evaluate.py`. Edit it to customize metrics used in the evolution process. Current set to return:
- **avg_btb**: Average "better than baseline" - percentage improvement over baseline (PRIMARY)
- **combined_score**: Score (...)
- **eval_time**: Total execution time in seconds
- **memory**: Total memory usage in KB

