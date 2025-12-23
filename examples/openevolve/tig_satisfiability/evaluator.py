import os
import subprocess
import math
from typing import List, Dict
import sys

# Add tig-evolve root to path to import tig module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from tig import build_algorithm, run_test

NUM_SEEDS = 5
TRACK_ID = "n_vars=100000,ratio=4150"
TIMEOUT = 60


def calculate_scores(qualities: List[float], times: List[float], memories: List[float]) -> Dict[str, float]:
    """
    Calculate normalized scores from raw metrics.
    """
    # Performance score: normalize by the highest achievable btb (~0.0014)
    # This represents ~0.14% improvement over baseline
    MAX_BTB = 1.0
    avg_btb = sum(qualities) / len(qualities)
    performance_score = avg_btb/MAX_BTB
    
    eval_time = sum(times) 
    # Logarithmic speed score: each order of magnitude faster = +0.1 score
    # 6000s -> 0, 600s -> 0.1, 60s -> 0.2, 6s -> 0.3, etc.
    if eval_time > 0:
        speed_score = max(0.0, min(1.0, 0.1 * math.log10(60.0 / eval_time)))
    else:
        speed_score = 1.0
    memory = sum(memories)
    if performance_score == 1.0:
        combined_score = (0.5*performance_score + 0.5*speed_score)
    else:
        combined_score = 0.5*performance_score
    
    return {
        "avg_btb": float(avg_btb),
        "combined_score": float(combined_score),
        "eval_time": float(eval_time),
        "memory": float(memory)
    }


def evaluate(program_path: str):
    try:
        abs_program_path = os.path.abspath(program_path)
        program_dir = os.path.dirname(abs_program_path)

        # Get the tig-evolve root directory (where algo-runner is located)
        tig_evolve_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
        
        original_dir = os.getcwd()
        os.chdir(tig_evolve_dir)

        # Copy algorithm code to algo-runner/src/algorithm/mod.rs
        with open(abs_program_path, "r") as src:
            content = src.read()
        algo_path = os.path.join(tig_evolve_dir, "algo-runner/src/algorithm/mod.rs")
        with open(algo_path, "w") as f:
            f.write(content)

        # Run build_algorithm
        try:
            build_algorithm()
        except subprocess.CalledProcessError as e:
            os.chdir(original_dir)
            return {"avg_btb": 0.0, "combined_score": 0.0, "eval_time": 0.0, "memory": 0.0, "error": f"Build failed: {e}"}
        
        # Run algorithm for each seed and collect results
        qualities = []
        times = []
        memories = []
        
        for seed in range(NUM_SEEDS):
            quality, time_seconds, memory = run_test(TRACK_ID, seed, timeout=TIMEOUT)
            if quality is not None and time_seconds is not None and memory is not None:
                # print(f"Seed: {seed}, Quality: {quality}, Time: {time_seconds}s, Memory: {memory}KB")
                qualities.append(float(quality) / 1000000.0)
                times.append(float(time_seconds))
                memories.append(float(memory))

        if program_dir in sys.path:
            sys.path.remove(program_dir)

        os.chdir(original_dir)

        if not qualities:
            return {"avg_btb": 0.0, "combined_score": 0.0, "eval_time": 0.0, "memory": 0.0, "error": "No successful evaluations"}

        scores = calculate_scores(qualities, times, memories)

        return {"avg_btb": scores["avg_btb"], "combined_score": scores["combined_score"], "eval_time": scores["eval_time"], "memory": scores["memory"]}
    except Exception as e:
        return {"avg_btb": 0.0, "combined_score": 0.0, "eval_time": 0.0, "memory": 0.0, "error": str(e)}
