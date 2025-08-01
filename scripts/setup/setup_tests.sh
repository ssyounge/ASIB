#!/usr/bin/env bash
# scripts/setup_tests.sh
# Install PyTorch and other dependencies required for running the unit tests.

set -e
python -m pip install --upgrade pip

# Install PyTorch CPU build by default
pip install --extra-index-url https://download.pytorch.org/whl/cpu torch torchvision

# Install remaining dependencies
pip install -r requirements.txt
pip install pytest
