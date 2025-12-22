#!/bin/bash
# Usage: bash examples/openevolve/run.sh <challenge>
# Example: bash examples/openevolve/run.sh knapsack
#          bash examples/openevolve/run.sh satisfiability
#          bash examples/openevolve/run.sh vehicle_routing
#
# This script should be run from the tig-evolve root folder

set -e  # Exit on any error

# --- Parse command line arguments ---
CHALLENGE="${1:-knapsack}"

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIG_EVOLVE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CHALLENGE_DIR="tig_${CHALLENGE}"
OPENEVOLVE_DIR="${TIG_EVOLVE_ROOT}/../openevolve"
VENV_DIR="${OPENEVOLVE_DIR}/.venv"
OPENEVOLVE_TAG="v0.2.23"

echo "=== OpenEvolve Setup Script ==="
echo "Challenge: ${CHALLENGE}"
echo "TIG-Evolve root: ${TIG_EVOLVE_ROOT}"
echo "OpenEvolve will be at: ${OPENEVOLVE_DIR}"

# --- Step 1: Clone openevolve if not present ---
if [ ! -d "${OPENEVOLVE_DIR}" ]; then
    echo ""
    echo ">>> Cloning openevolve repository..."
    git clone https://github.com/algorithmicsuperintelligence/openevolve.git "${OPENEVOLVE_DIR}"
    cd "${OPENEVOLVE_DIR}"
    git checkout "${OPENEVOLVE_TAG}"
    cd "${TIG_EVOLVE_ROOT}"
else
    echo ""
    echo ">>> openevolve already exists at ${OPENEVOLVE_DIR}"
fi

# --- Step 2: Create virtual environment if not present ---
if [ ! -d "${VENV_DIR}" ]; then
    echo ""
    echo ">>> Creating Python virtual environment..."
    python3 -m venv "${VENV_DIR}"
    echo "Virtual environment created"
else
    echo ""
    echo ">>> Virtual environment already exists at ${VENV_DIR}"
fi

# --- Step 3: Activate virtual environment ---
echo ""
echo ">>> Activating virtual environment..."
source "${VENV_DIR}/bin/activate"

# --- Step 4: Install openevolve and dependencies ---
if ! python3 -c "import openevolve" 2>/dev/null; then
    echo ""
    echo ">>> Installing openevolve and dependencies..."
    pip install --upgrade pip
    pip install requests  # Required by tig.py
    pip install openevolve
else
    echo ""
    echo ">>> openevolve already installed"
fi

# --- Step 5: Set API credentials ---
# OpenEvolve uses OPENAI_API_KEY environment variable
if [ -z "${OPENAI_API_KEY}" ]; then
    echo ""
    echo ">>> OPENAI_API_KEY not set. You can set it with:"
    echo "    export OPENAI_API_KEY='your-api-key'"
    echo ""
    echo "    For Google Gemini (default in config):"
    echo "    export OPENAI_API_KEY='your-google-ai-studio-api-key'"
    echo ""
    echo "Using placeholder - update before running experiments!"
    export OPENAI_API_KEY="${OPENAI_API_KEY:-placeholder_key}"
fi

echo ""
echo ">>> API Configuration:"
echo "    OPENAI_API_KEY: ${OPENAI_API_KEY:0:10}... (truncated)"

# --- Step 6: Initialize the challenge ---
echo ""
echo ">>> Initializing ${CHALLENGE} challenge..."
cd "${TIG_EVOLVE_ROOT}"
python3 tig.py init_challenge "${CHALLENGE}" --force

# --- Step 7: Run openevolve ---
echo ""
echo ">>> Running openevolve for ${CHALLENGE}..."

export TIG_EVOLVE_ROOT="${TIG_EVOLVE_ROOT}"

# Run from tig-evolve root directory
python3 ${OPENEVOLVE_DIR}/openevolve-run.py \
    "examples/openevolve/${CHALLENGE_DIR}/initial_program.rs" \
    "examples/openevolve/${CHALLENGE_DIR}/evaluator.py" \
    --config "examples/openevolve/${CHALLENGE_DIR}/config.yaml" \
    --iterations 3  # Only for testing purposes
