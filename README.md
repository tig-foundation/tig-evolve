# TIG-Evolve

**Hub for genetic coding agents for evolving and optimizing algorithms for [The Innovation Game (TIG)](https://tig.foundation)**

---

## Example Integrations with Evolution Frameworks

This repository hosts examples integrating with:

* [OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve)
* [CodeEvolve](https://github.com/inter-co/science-codeevolve) - Coming Soon
* [DeepEvolve](https://github.com/liugangcode/deepevolve) - Coming Soon

## Project Structure

```
tig-evolve/
├── tig.py                     # CLI tool for TIG algorithm management
├── algo-runner/               # Rust project for algorithm execution
│   ├── Cargo.toml
│   └── src/
│       ├── main.rs            # CLI runner (solve/eval commands)
│       ├── challenge/         # Challenge definition (from TIG)
│       └── algorithm/
│           └── mod.rs         # Algorithm code (evolved by LLMs)
└── examples/
    └── openevolve/            # OpenEvolve integration
        └── tig_knapsack/
            ├── run.sh         # One-click setup & run
            ├── initial_program.rs
            ├── evaluator.py
            └── config.yaml
```

## Quick Start

### OpenEvolve 

[See examples/openevolve/README.md](examples/openevolve/README.md#quick-start)

### CodeEvolve

Coming soon

### DeepEvolve

Coming soon

## Integrating with Other Evolution Frameworks

`tig.py` is the core tool. All of the commands available via CLI are also importable and usable from Python.

### CLI Usage

```
usage: tig.py [-h] {init_challenge,download_algorithm,build_algorithm,list_algorithms,test_algorithm} ...

TIG Evolve CLI Tool

positional arguments:
  {init_challenge,download_algorithm,build_algorithm,list_algorithms,test_algorithm}
                        Available commands
    init_challenge      Initialize a challenge
    download_algorithm  Download an algorithm
    build_algorithm     Build the algorithm
    list_algorithms     List available algorithms
    test_algorithm      Test the algorithm
```

### Python API

```python
from tig import *

# Initialize a challenge
init_challenge(challenge, force=True)

# List available algorithms
list_algorithms(challenge)  # returns list of algorithms

# Download an algorithm
download_algorithm(challenge, algorithm, force=True)  # downloads algorithm to algo-runner/src

# Build the algorithm
build_algorithm()  # compiles algo-runner

# Test the algorithm
run_test(track_id, seed, hyperparameters=None, timeout=60, debug=False)
```