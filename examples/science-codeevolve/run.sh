#!/bin/bash
# Usage: bash examples/science-codeevolve/run.sh <challenge_name>
# Example: bash examples/science-codeevolve/run.sh knapsack
# This script should be run from the tig-evolve root folder

set -e  # Exit on any error

# Get challenge name from argument (default to knapsack if not provided)
CHALLENGE_NAME="${1:-knapsack}"

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIG_EVOLVE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CODEEVOLVE_DIR="${TIG_EVOLVE_ROOT}/../science-codeevolve"
VENV_DIR="${CODEEVOLVE_DIR}/venv"
PYTHON_VERSION="3.13"  # codeevolve requires Python >=3.13.5

echo "=== CodeEvolve Setup Script ==="
echo "Challenge: ${CHALLENGE_NAME}"
echo "TIG-Evolve root: ${TIG_EVOLVE_ROOT}"
echo "CodeEvolve will be at: ${CODEEVOLVE_DIR}"

# --- Step 1: Clone science-codeevolve if not present ---
if [ ! -d "${CODEEVOLVE_DIR}" ]; then
    echo ""
    echo ">>> Cloning science-codeevolve repository..."
    git clone https://github.com/AoibheannTIG/science-codeevolve.git "${CODEEVOLVE_DIR}"
    # Checkout specific commit
    cd "${CODEEVOLVE_DIR}"
    git checkout 08c6e80
    cd "${TIG_EVOLVE_ROOT}"
else
    echo ""
    echo ">>> science-codeevolve already exists at ${CODEEVOLVE_DIR}"
fi

# --- Step 2: Install Python 3.13 if not present ---
if ! command -v python${PYTHON_VERSION} &> /dev/null; then
    echo ""
    echo ">>> Python ${PYTHON_VERSION} not found. Installing..."
    
    # Detect OS and install Python 3.13
    if [ -f /etc/debian_version ] || [ -f /etc/lsb-release ]; then
        # Ubuntu/Debian
        echo "Detected Ubuntu/Debian. Installing Python ${PYTHON_VERSION} from deadsnakes PPA..."
        sudo add-apt-repository -y ppa:deadsnakes/ppa
        sudo apt update
        sudo apt install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-venv python${PYTHON_VERSION}-dev
    elif [ -f /etc/redhat-release ]; then
        # RHEL/CentOS/Fedora
        echo "Detected RHEL/CentOS/Fedora. Please install Python ${PYTHON_VERSION} manually."
        echo "Try: sudo dnf install python${PYTHON_VERSION}"
        exit 1
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        echo "Detected macOS. Installing Python ${PYTHON_VERSION} via Homebrew..."
        if ! command -v brew &> /dev/null; then
            echo "Homebrew not found. Please install Homebrew first: https://brew.sh"
            exit 1
        fi
        brew install python@${PYTHON_VERSION}
    else
        echo "Unsupported OS. Please install Python ${PYTHON_VERSION} manually."
        echo "Download from: https://www.python.org/downloads/"
        exit 1
    fi
    
    # Verify installation
    if ! command -v python${PYTHON_VERSION} &> /dev/null; then
        echo "Error: Python ${PYTHON_VERSION} installation failed."
        exit 1
    fi
    echo "Python ${PYTHON_VERSION} installed successfully."
else
    echo ""
    echo ">>> Python ${PYTHON_VERSION} already installed"
fi

PYTHON_CMD="python${PYTHON_VERSION}"

# --- Step 3: Create virtual environment if not present ---
if [ ! -d "${VENV_DIR}" ]; then
    echo ""
    echo ">>> Creating Python virtual environment..."
    ${PYTHON_CMD} -m venv "${VENV_DIR}"
    echo "Virtual environment created with ${PYTHON_CMD}"
else
    echo ""
    echo ">>> Virtual environment already exists at ${VENV_DIR}"
fi

# --- Step 4: Activate virtual environment ---
echo ""
echo ">>> Activating virtual environment..."
source "${VENV_DIR}/bin/activate"

# --- Step 5: Install/upgrade pip and install codeevolve if needed ---
if ! command -v codeevolve &> /dev/null; then
    echo ""
    echo ">>> Installing codeevolve and dependencies..."
    pip install --upgrade pip
    pip install requests  # Required by tig.py
    pip install -e "${CODEEVOLVE_DIR}[dev,benchmarks]"
else
    echo ""
    echo ">>> codeevolve already installed"
fi

# --- Step 6: Set API credentials ---
# Check if API_KEY is already set, if not prompt or use placeholder
if [ -z "${API_KEY}" ]; then
    echo ""
    echo ">>> API_KEY not set. You can set it with:"
    echo "    export API_KEY='your-google-ai-studio-api-key'"
    echo ""
    echo "Using placeholder - update before running experiments!"
    export API_KEY="${API_BASE:-api_key}"
fi

export API_BASE="${API_BASE:-https://generativelanguage.googleapis.com/v1beta/openai/}"

echo ""
echo ">>> API Configuration:"
echo "    API_BASE: ${API_BASE}"
echo "    API_KEY: ${API_KEY:0:10}... (truncated)"

# --- Step 7: Initialize the challenge ---
echo ""
echo ">>> Initializing ${CHALLENGE_NAME} challenge..."
cd "${TIG_EVOLVE_ROOT}"
python3 tig.py init_challenge "${CHALLENGE_NAME}" --force

# # --- Step 8: Generate config.yaml from template ---
# echo ""
# echo ">>> Generating config.yaml..."
# bash "${SCRIPT_DIR}/generate_config.sh" "${CHALLENGE_NAME}"

# --- Step 9: Run codeevolve ---
echo ""
echo ">>> Running codeevolve..."

export TIG_EVOLVE_ROOT="${TIG_EVOLVE_ROOT}"

codeevolve \
    --inpt_dir="examples/science-codeevolve/tig_${CHALLENGE_NAME}/input/" \
    --cfg_path="examples/science-codeevolve/tig_${CHALLENGE_NAME}/configs/config.yaml" \
    --out_dir="examples/science-codeevolve/tig_${CHALLENGE_NAME}/experiments/config/" \
    --load_ckpt=0 \
    --terminal_logging