#!/bin/bash
# Training launcher for Parkinson's Project

set -e

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo "🚀 Parkinson's Project Training Launcher"
echo "=" * 50

# Forward all arguments to the main training script
exec "$PROJECT_ROOT/scripts/train.sh" "$@"