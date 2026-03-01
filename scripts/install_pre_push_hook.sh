#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
HOOK_PATH="${REPO_ROOT}/.git/hooks/pre-push"

if [[ ! -d "${REPO_ROOT}/.git" ]]; then
    echo "ERROR: ${REPO_ROOT} is not a git repository (missing .git)." >&2
    exit 1
fi

cat > "${HOOK_PATH}" <<'HOOK'
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
"${REPO_ROOT}/scripts/test_gate.sh"
HOOK

chmod +x "${HOOK_PATH}"

echo "Installed pre-push hook at ${HOOK_PATH}"
