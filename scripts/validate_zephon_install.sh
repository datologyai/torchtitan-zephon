#!/usr/bin/env bash
# Validate TorchTitan against an installed Zephon distribution in a clean venv.

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
validation_dir=$(mktemp -d "${TMPDIR:-/tmp}/torchtitan-zephon-validation.XXXXXX")
trap 'rm -rf "${validation_dir}"' EXIT

uv venv --python "${PYTHON_VERSION:-3.12}" "${validation_dir}/venv"
python_path="${validation_dir}/venv/bin/python"

uv pip install --python "${python_path}" -e "${repo_root}[dev]"
if [[ -n "${ZEPHON_WHEEL:-}" ]]; then
    uv pip install --python "${python_path}" \
        "${ZEPHON_WHEEL}" \
        "transformers>=4.44.0"
else
    uv pip install --python "${python_path}" \
        -r "${repo_root}/requirements-zephon.txt"
fi

cd "${repo_root}"
"${python_path}" -c \
    'import zephon; print(f"Validating Zephon {zephon.__version__} from {zephon.__file__}")'
HF_HOME="${validation_dir}/hf-cache" "${python_path}" -m pytest \
    tests/unit_tests/test_zephon_dataloader_override.py -q
HF_HOME="${validation_dir}/hf-cache" "${python_path}" \
    examples/zephon/elastic_resume_demo.py
