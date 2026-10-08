#!/usr/bin/env bash

set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
submodule="$repo_root/tools/Freeze-Omni"
patch_dir="$repo_root/patches/Freeze-Omni"

git -C "$repo_root" submodule update --init -- tools/Freeze-Omni

if ! git -C "$submodule" diff --quiet || ! git -C "$submodule" diff --cached --quiet; then
	echo "Error: tools/Freeze-Omni must have no tracked changes before applying patches." >&2
	exit 1
fi

shopt -s nullglob
patches=("$patch_dir"/*.patch)
if [[ ${#patches[@]} -eq 0 ]]; then
	echo "No Freeze-Omni patches found."
	exit 0
fi

for patch in "${patches[@]}"; do
	echo "Applying $(basename "$patch")"
	git -C "$submodule" apply --whitespace=nowarn --check "$patch"
	git -C "$submodule" apply --whitespace=nowarn "$patch"
done
