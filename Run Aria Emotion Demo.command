#!/bin/zsh
set -e

PROJECT_DIR="${0:A:h}"
ARIA_ENV="/Users/dinithidissanayake/Documents/Aria 2/.venv-gen2"

export PATH="$ARIA_ENV/bin:$PATH"
cd "$PROJECT_DIR"

"$ARIA_ENV/bin/python" src/live_audio_emotion.py --model-dir models
