#!/bin/bash
# Build a folder of symlinks to the documents inside a set of code repos, to
# index as one note mount (VAULT_MCP_NOTE_MOUNTS / note_mounts).
#
# A repo is mostly code, and walking all of it for the few documents in it is
# slow and noisy. This links only what reads as documentation — each repo's
# docs/, plans/ and specs/ folders and its top-level .md files — under
# <out>/<repo>/, which the indexer then walks like any other folder (it follows
# symlinks). Re-run it when a repo gains its first docs folder; files added to
# a folder already linked show up on their own.
#
# Usage: link-repo-docs.sh <out-dir> <repos-dir> [<repos-dir> ...]
set -euo pipefail
shopt -s nullglob

out="${1:?out dir}"; shift
[ "$#" -ge 1 ] || { echo "usage: $0 <out-dir> <repos-dir> [...]" >&2; exit 2; }

rm -rf "$out"
mkdir -p "$out"
for root in "$@"; do
  for repo in "$root"/*/; do
    repo="${repo%/}"
    name="$(basename "$repo")"
    [ -e "$out/$name" ] && continue
    links=()
    for sub in docs plans specs; do
      [ -d "$repo/$sub" ] && links+=("$repo/$sub")
    done
    for file in "$repo"/*.md; do
      [ -f "$file" ] && links+=("$file")
    done
    [ "${#links[@]}" -eq 0 ] && continue
    mkdir -p "$out/$name"
    for target in "${links[@]}"; do
      ln -s "$target" "$out/$name/$(basename "$target")"
    done
  done
done
echo "linked $(find "$out" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ') repos into $out"
