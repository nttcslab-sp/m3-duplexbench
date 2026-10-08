# PersonaPlex patches

`tools/personaplex` tracks the official NVIDIA repository at commit
`3428dfd95309a7f3c84fd93259ded0f810d1ff91`.

Repository-specific changes are stored here instead of being committed to the
submodule. Apply them from the repository root with:

```bash
git submodule update --init tools/personaplex
bash scripts/apply_personaplex_patches.sh
```

The script requires a clean submodule worktree and applies patches in filename
order.

The previous internal fork is preserved locally at `tools/personaplex_legacy`,
with its original internal remote configured as `origin`.

When updating PersonaPlex, first update the submodule to the desired official
commit, then verify the existing patches against that commit and regenerate any
patches that no longer apply before committing the updated submodule gitlink and
patch series.
