# ===--------------------------------------------------------------------------------------===#
#
# Part of the CodeEvolve Project, under the Apache License v2.0.
# See https://github.com/inter-co/science-codeevolve/blob/main/LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0
#
# ===--------------------------------------------------------------------------------------===#
#
# This file implements the evaluator for the TIG Vehicle Routing problem.
# It builds and evaluates Rust-based vehicle routing algorithms.
#
# ===--------------------------------------------------------------------------------------===#

import os
import sys
import math
import json
from typing import List, Dict

# tig-evolve root - use environment variable or default to absolute path
# This is needed because CodeEvolve copies this script to a temp directory
TIG_EVOLVE_ROOT = os.environ.get("TIG_EVOLVE_ROOT", "/root/tig-evolve")
ALGO_RUNNER_DIR = os.path.join(TIG_EVOLVE_ROOT, "algo-runner")

# Add tig-evolve root to path to import tig.py functions
sys.path.insert(0, TIG_EVOLVE_ROOT)
from tig import build_algorithm, run_test

NUM_SEEDS = 4
TRACK_ID = "n_nodes=600"
TIMEOUT = 600 


def performance_scale(x: float, max_btb: float) -> float:
    """Smoothly scale performance based on better-than-baseline metric."""
    if max_btb <= 0:
        return 0.0

    numerator = math.exp(16.0 * x) - 1.0
    denominator = math.exp(16.0 * max_btb) - 1.0

    if denominator == 0.0:
        return 0.0

    return max(0.0, min(numerator / denominator, 1.0))


def calculate_scores(qualities: List[float], times: List[float], memories: List[float]) -> Dict[str, float]:
    """Calculate normalized scores from raw metrics."""
    # Performance score: normalize by the highest achievable btb (~0.0014)
    MAX_BTB = 0.2
    avg_btb = sum(qualities) / len(qualities)
    if avg_btb < 0:
        performance_score = (0.5 + float(avg_btb)) / 10.0
    else:
        performance_score = avg_btb/MAX_BTB
    
    eval_time = sum(times) 
    # Logarithmic speed score
    if eval_time > 0:
        speed_score = max(0.0, min(1.0, 0.1 * math.log10(600.0 / eval_time)))
    else:
        speed_score = 1.0
    memory = sum(memories)
    combined_score = (0.9 * performance_score) + (0.1 * speed_score)
    
    return {
        "avg_btb": float(avg_btb),
        "combined_score": float(combined_score),
        "eval_time": float(eval_time),
        "memory": float(memory)
    }


def evaluate(program_path: str, results_path: str) -> None:
    """
    Evaluate a TIG Vehicle Routing algorithm.
    
    Args:
        program_path: Path to the Rust source file containing the algorithm
        results_path: Path where JSON results should be written
    """
    try:
        abs_program_path = os.path.abspath(program_path)
        original_dir = os.getcwd()

        # Change to tig-evolve root so tig.py functions work correctly
        os.chdir(TIG_EVOLVE_ROOT)

        # Copy algorithm code to algo-runner/src/algorithm/mod.rs
        with open(abs_program_path, "r") as src:
            content = src.read()
        algo_path = os.path.join(ALGO_RUNNER_DIR, "src/algorithm/mod.rs")
        with open(algo_path, "w") as f:
            f.write(content)

        # Build the algorithm using tig.py function
        try:
            build_algorithm()
        except Exception as e:
            os.chdir(original_dir)
            with open(results_path, "w") as f:
                json.dump({
                    "avg_btb": 0.0,
                    "combined_score": 0.0,
                    "eval_time": 0.0,
                    "memory": 0.0,
                    "error": f"Build failed: {str(e)}"
                }, f, indent=4)
            return

        # Run algorithm for each seed and collect results
        qualities = []
        times = []
        memories = []
        
        for seed in range(NUM_SEEDS):
            quality, time_seconds, memory = run_test(
                track_id=TRACK_ID,
                seed=seed,
                timeout=TIMEOUT,
                debug=False
            )
            if quality is not None and time_seconds is not None and memory is not None:
                qualities.append(float(quality) / 1000000.0)
                times.append(float(time_seconds))
                memories.append(float(memory))

        os.chdir(original_dir)

        if not qualities:
            with open(results_path, "w") as f:
                json.dump({
                    "avg_btb": 0.0,
                    "combined_score": 0.0,
                    "eval_time": 0.0,
                    "memory": 0.0,
                    "error": "No successful evaluations"
                }, f, indent=4)
            return

        scores = calculate_scores(qualities, times, memories)
        
        with open(results_path, "w") as f:
            json.dump({
                "avg_btb": scores["avg_btb"],
                "combined_score": scores["combined_score"],
                "eval_time": scores["eval_time"],
                "memory": scores["memory"]
            }, f, indent=4)
            
    except Exception as e:
        with open(results_path, "w") as f:
            json.dump({
                "avg_btb": 0.0,
                "combined_score": 0.0,
                "eval_time": 0.0,
                "memory": 0.0,
                "error": str(e)
            }, f, indent=4)


if __name__ == "__main__":
    program_path = sys.argv[1]
    results_path = sys.argv[2]
    
    evaluate(program_path, results_path)
