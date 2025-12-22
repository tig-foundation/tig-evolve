# Evolving Algorithm for TIG with OpenEvolve

This example uses [CodeEvolve](https://github.com/inter-co/science-codeevolve.git) to optimize algorithms for TIG.

## Supported TIG Challenges

| Challenge | Description |
|-----------|-------------|
| **knapsack** | Quadratic knapsack problem |
| **vehicle_routing** | Vehicle routing problem with time windows|
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
bash examples/science-codeevolve/run.sh knapsack
```

The script will:
1. Clone CodeEvolve
2. Create a Python virtual environment
3. Install dependencies
4. Initialize the knapsack challenge
5. Start the evolution process

### Testing the Best Program

After running CodeEvolve, the best program is saved in `best` folder. To test this:

```bash
# Copy the best program to algo-runner
cp examples/science-codeevolve/tig_knapsack/experiments/config/0/best_sol.rs algo-runner/src/algorithm/mod.rs

python3 tig.py build_algorithm
python3 tig.py test_algorithm "n_items=500,density=25" --tests 1000
```

Alternatively, this can be done programmatically in Python:

```bash
cd examples/science-codeevolve/tig_knapsack
python3 -c "
from evaluator import evaluate
result = evaluate('experiments/config/0/best_sol.rs')
print('Result:', result)
"
```

## Configuring OpenEvolve

Edit `config.yaml` to customize:

- **LLM settings**: Model selection, API base URL, temperature
- **Evolution parameters**: Population size, islands, migration rates
- **Prompts**: System message with optimization guidance
- **Evaluation**: Timeout, parallel evaluations

The default config uses Google Gemini models. To use other providers, update `llm.api_base` and `llm.primary_model`/`llm.secondary_model`.

## Configuring Evaluation Metrics

Edit `evaluator.py` to customize metrics used in the evolution process. Current set to return:
- **avg_btb**: Average "better than baseline" - percentage improvement over baseline (PRIMARY)
- **combined_score**: Score (...)
- **eval_time**: Total execution time in seconds
- **memory**: Total memory usage in KB

