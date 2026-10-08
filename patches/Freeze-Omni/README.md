# Freeze-Omni patches

`tools/Freeze-Omni` tracks the official VITA-MLLM repository at commit
`163a24880e533b2a07038fb8dcfe02dbbb8457e6`.

Repository-specific changes are stored here instead of being committed to the
submodule. Apply them from the repository root with:

```bash
git submodule update --init tools/Freeze-Omni
bash scripts/apply_freeze_omni_patches.sh
```

The script requires no tracked changes in the submodule and applies patches in
filename order. Untracked model weights, checkpoints, and certificates are
allowed.

When updating Freeze-Omni, first update the submodule to the desired official
commit. Then verify the existing patches against that commit and regenerate any
patches that no longer apply before committing the updated gitlink and patch
series.
