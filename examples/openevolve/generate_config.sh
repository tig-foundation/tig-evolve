#!/bin/bash
# Generate config.yaml from template by inserting the challenge README
#
# Usage: ./generate_config.sh <challenge_name>
# Example: ./generate_config.sh knapsack
#          ./generate_config.sh satisfiability
#          ./generate_config.sh vehicle_routing
#
# This script reads config.yaml.template and replaces the
# {{challenge_readme}} placeholder with the contents of the challenge README.

set -e

# Get challenge name from argument (default to knapsack if not provided)
CHALLENGE_NAME="${1:-knapsack}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="$SCRIPT_DIR/tig_${CHALLENGE_NAME}/config.yaml.template"
OUTPUT="$SCRIPT_DIR/tig_${CHALLENGE_NAME}/config.yaml"
CHALLENGE_DIR="$SCRIPT_DIR/../../algo-runner/src/challenge"

# Map challenge name to README file
case "$CHALLENGE_NAME" in
    knapsack)
        README="$CHALLENGE_DIR/README.md"
        ;;
    satisfiability)
        README="$CHALLENGE_DIR/README_sat.md"
        ;;
    vehicle_routing)
        README="$CHALLENGE_DIR/README_vrp.md"
        ;;
    *)
        # Default: try README_<challenge>.md, then README.md
        if [[ -f "$CHALLENGE_DIR/README_${CHALLENGE_NAME}.md" ]]; then
            README="$CHALLENGE_DIR/README_${CHALLENGE_NAME}.md"
        else
            README="$CHALLENGE_DIR/README.md"
        fi
        ;;
esac

echo "=== Generating config.yaml ==="
echo "Challenge: ${CHALLENGE_NAME}"
echo "Template: ${TEMPLATE}"
echo "README: ${README}"

# Check that required files exist
if [[ ! -f "$TEMPLATE" ]]; then
    echo "Error: Template file not found: $TEMPLATE"
    exit 1
fi

if [[ ! -f "$README" ]]; then
    echo "Warning: README file not found: $README"
    echo "Creating config without README content..."
    README_CONTENT="    (No challenge README available)"
else
    # Read README content and indent each line with 4 spaces (for YAML block scalar)
    README_CONTENT=$(sed 's/^/    /' "$README")
fi

# Create a temporary file for safe replacement
TMP_FILE=$(mktemp)

# Read template and replace placeholder
# Using awk for reliable multiline replacement
awk -v readme="$README_CONTENT" '{
    if ($0 ~ /\{\{challenge_readme\}\}/) {
        print readme
    } else {
        print
    }
}' "$TEMPLATE" > "$TMP_FILE"

# Move temp file to output
mv "$TMP_FILE" "$OUTPUT"

echo "Generated: $OUTPUT"
