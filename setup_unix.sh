#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_root"

python3.11 -m venv .venv
python_bin="$project_root/.venv/bin/python"

"$python_bin" -m pip install --upgrade pip setuptools wheel
# CPU is the portable default. macOS wheels are served from PyPI; Linux CPU
# wheels use PyTorch's CPU index. For CUDA/ROCm, use the official selector.
if [[ "$(uname -s)" == "Darwin" ]]; then
  "$python_bin" -m pip install 'torch==2.12.1'
else
  "$python_bin" -m pip install 'torch==2.12.1' --index-url https://download.pytorch.org/whl/cpu
fi
"$python_bin" -m pip install -r requirements.txt
"$python_bin" -m pip install -e packages/metamaterial_envs
"$python_bin" scripts/configure_paths.py --create
"$python_bin" scripts/verify_install.py --quick

printf '\nEnvironment is ready. Activate with: source .venv/bin/activate\n'
