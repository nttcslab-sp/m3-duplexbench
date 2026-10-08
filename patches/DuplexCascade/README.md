# DuplexCascade patches

`tools/DuplexCascade` is pinned to an upstream commit. Local changes are kept as
patches in this directory instead of being committed to the submodule.

After cloning the parent repository, initialize the submodule and apply the
patches with:

```bash
git submodule update --init tools/DuplexCascade
bash scripts/apply_duplexcascade_patches.sh
```

The script requires a clean submodule worktree and applies the patches in
filename order. Applying them intentionally makes the submodule worktree dirty;
the parent repository continues to track only the upstream commit.

When updating DuplexCascade, first move the submodule to the desired upstream
commit, then verify that all patches still apply:

```bash
git -C tools/DuplexCascade apply --unidiff-zero --check \
  ../../patches/DuplexCascade/0001-Add-SSL-certificate-options.patch
```

If this check fails, rebase the local change onto the new upstream revision and
regenerate the corresponding patch before committing the updated submodule
pointer.
