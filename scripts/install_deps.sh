#!/bin/bash
# Reinstall SAM2 and GroundingDINO (they may have been in /private/tmp/ which is volatile).
# Run this after a reboot or if import errors appear.
set -e

echo "Installing SAM2..."
pip install git+https://github.com/facebookresearch/sam2.git --break-system-packages

echo "Checking GroundingDINO..."
python3 -c "import groundingdino" 2>/dev/null || \
  pip install git+https://github.com/IDEA-Research/GroundingDINO.git --break-system-packages

echo "Done."
