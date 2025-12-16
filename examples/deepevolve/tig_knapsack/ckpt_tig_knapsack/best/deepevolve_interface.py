import json
import math
import os
import shutil
import subprocess
import traceback
from pathlib import Path


def detect_repo_root(marker: str = "tig.py") -> Path:
    """Walk up from this file until we find the repo marker (tig.py)."""
    for parent in Path(__file__).resolve().parents:  # DEBUG: Ensure the search for parent directories is accurate
        # DEBUG: Check if tig.py exists in the candidate path
        candidate = parent if (parent / marker).exists() else None
        if candidate:
            return candidate
    # DEBUG: Providing clearer error to help with debugging
    raise FileNotFoundError(  # DEBUG: Provide a clearer error message for debugging
        f"Could not locate repository root containing {marker}. "
        f"Searched up to: {Path(__file__).resolve().parents}. "
        "Set TIG_REPO_ROOT to override."
    )
    # DEBUG: Adjusted indentation to fix unexpected indent error
    if candidate:
        return candidate


def resolve_repo_root() -> Path:
    """Resolve the TIG repo root via env override or automatic detection."""
    env_path = os.getenv("TIG_REPO_ROOT")
    if env_path:
        return Path(env_path).expanduser().resolve()
    return detect_repo_root()


# Absolute path to the TIG repo (auto-detected unless TIG_REPO_ROOT overrides)
REPO_ROOT = resolve_repo_root()
ALGO_RUNNER = REPO_ROOT / "algo-runner"

# Track to evaluate; override with TIG_TRACK_ID env if needed
TRACK_ID = os.getenv("TIG_TRACK_ID", "n_items=500,density=5")

# Quick evaluation defaults
NUM_TESTS = int(os.getenv("TIG_NUM_TESTS", "10"))
TIMEOUT = int(os.getenv("TIG_TIMEOUT", "60"))
QUALITY_PRECISION = 1_000_000  # Matches algo-runner/src/lib.rs
MAX_BTB = 0.001


def performance_scale(x: float, max_btb: float) -> float:
    """Smoothly scale performance based on better-than-baseline metric."""
    if max_btb <= 0:
        return 0.0

    numerator = math.exp(3000.0 * x) - 1.0
    denominator = math.exp(3000.0 * max_btb) - 1.0

    if denominator == 0.0:
        return 0.0

    return max(0.0, min(numerator / denominator, 1.0))


def run_cmd(cmd, cwd):
    """Run a command and return (ok, stdout, stderr)."""
    res = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    return res.returncode == 0, res.stdout, res.stderr


def parse_metrics(stdout: str):
    """
    Parse tig.py test_algorithm output lines like:
    Seed: 0, Quality: <q>, Time: <t>, Memory: <m>KB
    """
    quality = None
    time_s = None
    mem_kb = None
    for line in stdout.splitlines():
        if "Quality:" in line:
            parts = line.split(",")
            for part in parts:
                if "Quality:" in part:
                    try:
                        quality = float(part.split(":")[1].strip())
                    except Exception:
                        quality = None
                if "Time:" in part:
                    try:
                        time_s = float(part.split(":")[1].strip())
                    except Exception:
                        time_s = None
                if "Memory:" in part:
                    try:
                        mem_kb = int(
                            part.split(":")[1]
                            .strip()
                            .replace("KB", "")
                            .strip()
                        )
                    except Exception:
                        mem_kb = None
    return quality, time_s, mem_kb


def deepevolve_interface():
    try:
        # Locate evolved Rust sources in the temp workspace
        src_algo = Path(__file__).resolve().parent / "algo-runner" / "src" / "algorithm"
        if not src_algo.exists():
            return False, f"Missing evolved Rust sources at {src_algo}"

        dst_algo = ALGO_RUNNER / "src" / "algorithm"
        if dst_algo.exists():
            shutil.rmtree(dst_algo)
        shutil.copytree(src_algo, dst_algo)

        # DEBUG: Ensured that the path to tig.py is correctly set in the REPO_ROOT and accounted for missing file
        tig_path = REPO_ROOT / "tig.py"
        if not tig_path.exists():  # DEBUG: Added a check for tig.py existence
            return False, f"Missing tig.py at {tig_path}"
        ok, out, err = run_cmd(["python", str(tig_path), "build_algorithm"], cwd=REPO_ROOT)
        if not ok:  # DEBUG: Check if the build process failed
            return False, f"build_algorithm failed\nstdout:\n{out}\nstderr:\n{err}. Check if tig.py is properly located at {tig_path}"

        cmd = [
            "python",
            "tig.py",
            "test_algorithm",
            TRACK_ID,
            "--tests",
            str(NUM_TESTS),
            "--timeout",
            str(TIMEOUT),
        ]
        ok, out, err = run_cmd(cmd, cwd=REPO_ROOT)
        if not ok:
            return False, f"test_algorithm failed\nstdout:\n{out}\nstderr:\n{err}"

        quality, time_s, mem_kb = parse_metrics(out)
        if quality is None:
            return False, f"Could not parse quality from output:\n{out}"

        quality_normalized = quality / QUALITY_PRECISION
        scaled_quality = performance_scale(quality_normalized, MAX_BTB)

        metrics = {
            "combined_score": scaled_quality,
            "quality": scaled_quality,
            "raw_quality": quality_normalized,
            "time_seconds": time_s,
            "memory_kb": mem_kb,
        }
        return True, metrics

    except Exception:
        return False, traceback.format_exc()


