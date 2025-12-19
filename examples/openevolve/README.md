# Evolving Algorithm for TIG with OpenEvolve

This example uses [OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve) to optimize algorithms for TIG.

## Supported TIG Challenges

| Challenge | Description |
|-----------|-------------|
| **knapsack** | Quadratic knapsack problem |

## Requirements

- Python 3.13+
- Rust (install from [rustup.rs](https://rustup.rs))
- API Key for OpenAI API provider (we recommend [Google AI Studio](https://aistudio.google.com) for Gemini)

## Quick Start

The `run.sh` script handles all setup automatically:

```bash
# Set your API key (example uses Google Gemini API by default)
export OPENAI_API_KEY="your-google-ai-studio-api-key"

# Run the script from tig-evolve root
bash examples/openevolve/tig_knapsack/run.sh
```

The script will:
1. Clone OpenEvolve (v0.2.23)
2. Create a Python virtual environment
3. Install dependencies
4. Initialize the knapsack challenge
5. Start the evolution process

### Testing the Best Program

After running OpenEvolve, the best program is saved in `best` folder. To test this:

```bash
# Copy the best program to algo-runner
cp examples/openevolve/tig_knapsack/openevolve_output/best/best_program.rs algo-runner/src/algorithm/mod.rs

python3 tig.py build_algorithm
python3 tig.py test_algorithm "n_items=500,density=25" --tests 1000
```

Alternatively, this can be done programmatically in Python:

```bash
cd examples/openevolve/tig_knapsack
python3 -c "
from evaluator import evaluate
result = evaluate('openevolve_output/best/best_program.rs')
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


## Viewing Checkpoints

OpenEvolve saves checkpoints during evolution, including program data and prompts.

To view a checkpoint, use `view_prompt.py`:

```bash
python3 view_prompt.py \
    examples/openevolve/tig_knapsack/openevolve_output/checkpoints/<checkpoint_number>/programs/<program_id>.json
```

**Note:** Enable `include_prompts: true` in `config.yaml` to save prompts. Generation 0 programs won't have prompts as they represent the initial program.
