#!/bin/bash
# Recreate the PyTorch venv after HPC restart.
# Run from: /l/users/muhammad.airlangga/robust-material
set -e
VENV="$(dirname "$0")/venv"
python3 -m venv "$VENV"
"$VENV/bin/pip" install --quiet torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
"$VENV/bin/pip" install --quiet numpy pandas matplotlib scikit-learn tqdm einops
chmod +x "$VENV/bin/run_baseline.sh"
echo "Venv ready at $VENV"
