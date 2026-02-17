#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
VENV_PYTHON="${REPO_ROOT}/.venv/bin/python"

# Preserve caller cwd before forcing repo-root execution.
export ABLETON_MCP_LAUNCH_CWD="${PWD}"

cd "${REPO_ROOT}"

if [[ -x "${VENV_PYTHON}" ]]; then
    PYTHON_BIN="${VENV_PYTHON}"
else
    echo "ERROR: ${VENV_PYTHON} was not found. Falling back to python3 from PATH." >&2
    if ! command -v python3 >/dev/null 2>&1; then
        echo "ERROR: python3 is not available. Create ${REPO_ROOT}/.venv or install python3." >&2
        exit 1
    fi
    PYTHON_BIN="$(command -v python3)"
fi

if "${PYTHON_BIN}" "${REPO_ROOT}/MCP_Server/run_server.py" --smoke; then
    echo "OK"
    exit 0
fi

exit 1
