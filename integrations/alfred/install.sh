#!/bin/bash
# Copy the workflow into Alfred's preferences and point it at this repo's
# interpreter. Alfred does not follow symlinks into a workflow folder, so the
# files are copied; re-run after editing them.
#
# Usage: install.sh [<Alfred.alfredpreferences dir>]
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
python="$repo/.venv/bin/python"
[ -x "$python" ] || { echo "no interpreter at $python — run scripts/setup.sh first" >&2; exit 1; }

prefs="${1:-}"
if [ -z "$prefs" ]; then
  sync="$(defaults read com.runningwithcrayons.Alfred-Preferences syncfolder 2>/dev/null || true)"
  sync="${sync/#\~/$HOME}"
  prefs="${sync:-$HOME/Library/Application Support/Alfred}/Alfred.alfredpreferences"
fi
[ -d "$prefs/workflows" ] || { echo "no Alfred workflows folder at $prefs/workflows" >&2; exit 1; }

dest="$prefs/workflows/user.workflow.6C1B7E52-3F0A-4C57-9C0E-FF00000000FF"
mkdir -p "$dest"
cp "$here/info.plist" "$here/ff.sh" "$dest/"
printf '%s\n' "$python" > "$dest/python-path"
echo "installed to $dest"
