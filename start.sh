#!/usr/bin/env bash
cd "$(dirname "$0")"
[ -d .venv ] || python3 -m venv .venv || exit 1
source .venv/bin/activate
pip install -q --upgrade pip
if [ "$(uname)" = "Linux" ]; then pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu; fi
pip install -r requirements.txt || exit 1
pip install pillow-heif || true
python app.py
