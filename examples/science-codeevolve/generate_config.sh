#!/bin/bash
# Generate config.yaml from template by inserting the challenge README
#
# Usage: ./generate_config.sh <challenge_name>
# Example: ./generate_config.sh knapsack
#
# This script reads configs/config.yaml.template and replaces the
# {{CHALLENGE_README}} placeholder with the contents of the challenge README.

set -e

# Get challenge name from argument (default to knapsack if not provided)
CHALLENGE_NAME="${1:-knapsack}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="$SCRIPT_DIR/tig_${CHALLENGE_NAME}/configs/config.yaml.template"
OUTPUT="$SCRIPT_DIR/tig_${CHALLENGE_NAME}/configs/config.yaml"
README="$SCRIPT_DIR/../../algo-runner/src/challenge/README.md"

echo "=== Generating config.yaml ==="
echo "Challenge: ${CHALLENGE_NAME}"

# Check that required files exist
if [[ ! -f "$TEMPLATE" ]]; then
    echo "Error: Template file not found: $TEMPLATE"
    exit 1
fi

if [[ ! -f "$README" ]]; then
    echo "Error: README file not found: $README"
    exit 1
fi

# Read README content and indent each line with 2 spaces (for YAML block scalar)
README_CONTENT=$(sed 's/^/  /' "$README")

# Create a temporary file for safe replacement
TMP_FILE=$(mktemp)

# Read template and replace placeholder
# Using awk for reliable multiline replacement
awk -v readme="$README_CONTENT" '{
    if ($0 ~ /\{\{CHALLENGE_README\}\}/) {
        # Get the indentation of the placeholder line
        match($0, /^[[:space:]]*/)
        indent = substr($0, RSTART, RLENGTH)
        print readme
    } else {
        print
    }
}' "$TEMPLATE" > "$TMP_FILE"

# Move temp file to output
mv "$TMP_FILE" "$OUTPUT"

echo "Generated: $OUTPUT"
