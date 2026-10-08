# Moshi patches

`tools/moshi` tracks the official Kyutai repository at commit
`6d14a61994f38b282ae75bc4e9c3dcc4d35e7183`. The patches in this directory
reproduce the seven commits that were previously maintained in the internal
Moshi fork.

After cloning the parent repository, initialize the submodule and apply the
patches with:

```bash
git submodule update --init tools/moshi
bash scripts/apply_moshi_patches.sh
```

The script requires a clean submodule worktree and applies patches in filename
order. Applying them intentionally makes the submodule worktree dirty; the
parent repository continues to track only the official upstream commit.

The previous internal fork is preserved locally at `tools/moshi_legacy` and is
excluded from the parent repository by `.gitignore`.

When updating Moshi, first move the submodule to the desired official commit,
then verify and refresh the patches as needed before committing the updated
