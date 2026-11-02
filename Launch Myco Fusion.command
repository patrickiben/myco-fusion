#!/bin/zsh
set -e
cd "${0:A:h}"
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
if [[ -x .venv/bin/python ]]; then
  exec .venv/bin/python -m myco_fusion serve --open
else
  exec python3 -m myco_fusion serve --open
fi
