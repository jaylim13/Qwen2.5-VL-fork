#!/bin/bash
# Web application launcher for Parkinson's Project

set -e

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo "🌐 Parkinson's Project Web App Launcher"
echo "=" * 50

# Forward all arguments to the web app in parkinson_proj
exec python "$PROJECT_ROOT/parkinson_proj/web_application/web_interface/app.py" "$@"