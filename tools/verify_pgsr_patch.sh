#!/usr/bin/env bash
set -euo pipefail

repo=${1:?PGSR repository path is required}
patch=${2:?Patch path is required}
base=de24f1a38b350387e8d8fe381b2cd70c1ae946e7
tmp=$(mktemp -d /tmp/pgsr_patch_verify.XXXXXX)

cleanup() {
  git -C "$repo" worktree remove --force "$tmp" >/dev/null 2>&1 || true
  case "$tmp" in
    /tmp/pgsr_patch_verify.*) rm -rf -- "$tmp" ;;
    *) echo "Refusing to remove unexpected temporary path: $tmp" >&2 ;;
  esac
}
trap cleanup EXIT

git -C "$repo" worktree add --detach "$tmp" "$base" >/dev/null
git -C "$tmp" apply --check --whitespace=nowarn "$patch"
git -C "$tmp" apply --whitespace=nowarn "$patch"
python3 -m py_compile \
  "$tmp/gaussian_renderer/__init__.py" \
  "$tmp/scene/gaussian_model.py" \
  "$tmp/train.py" \
  "$tmp/utils/projective_validity.py"

for file in \
  gaussian_renderer/__init__.py \
  scene/gaussian_model.py \
  train.py \
  utils/projective_validity.py; do
  patched=$(sha256sum "$tmp/$file" | awk '{print $1}')
  active=$(sha256sum "$repo/$file" | awk '{print $1}')
  if [[ "$patched" != "$active" ]]; then
    echo "HASH_MISMATCH $file patched=$patched active=$active" >&2
    exit 1
  fi
  echo "MATCH $file $patched"
done

echo "PATCH_REPLAY=passed"
