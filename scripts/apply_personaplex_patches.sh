#!/usr/bin/env bash

set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
submodule="$repo_root/tools/personaplex"
patch_dir="$repo_root/patches/personaplex"

git -C "$repo_root" submodule update --init -- tools/personaplex

if [[ -n $(git -C "$submodule" status --porcelain) ]]; then
	echo "Error: tools/personaplex must be clean before applying patches." >&2
	exit 1
fi

shopt -s nullglob
patches=("$patch_dir"/*.patch)
if [[ ${#patches[@]} -eq 0 ]]; then
	echo "No PersonaPlex patches found."
	exit 0
fi

for patch in "${patches[@]}"; do
	echo "Applying $(basename "$patch")"
	git -C "$submodule" apply --whitespace=nowarn --check "$patch"
	git -C "$submodule" apply --whitespace=nowarn "$patch"
done
