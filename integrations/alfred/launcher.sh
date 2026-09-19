#!/bin/bash
# Alfred's entry point. install.sh writes the interpreter's path beside this
# file, because a workflow runs from Alfred's own folder, not from the repo.
here="$(cd "$(dirname "$0")" && pwd)"
python="$(cat "$here/python-path" 2>/dev/null)"
if [ ! -x "$python" ]; then
  echo '{"items":[{"title":"vault-mcp launcher is not installed","subtitle":"Run integrations/alfred/install.sh from the vault-mcp repo","valid":false}]}'
  exit 0
fi
exec "$python" -m vault_mcp.launcher "$@"
