#!/usr/bin/env bash
# Build every kernel-builder kernel under src/kops/<name>/ on the Spark via nix.
# Each kernel dir must be a git repo (nix flakes require it); `build-and-copy` writes build/ in place.
# Run ON THE SPARK after scripts/transfer.sh. Usage: scripts/build_kernels.sh [<name> ...]
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NAMES=("$@")

for d in "$ROOT"/src/kops/*/; do
  [ -f "$d/build.toml" ] || continue
  name="$(basename "$d")"
  if [ ${#NAMES[@]} -gt 0 ] && [[ ! " ${NAMES[*]} " == *" $name "* ]]; then continue; fi
  echo "=== building $name ==="
  (
    cd "$d"
    # source-only git tree so nix flake + build-and-copy stay clean
    printf 'build/\n.venv/\nresult\nCMakeLists.txt\ncmake/\n*.cmake\nsetup.py\npyproject.toml\ntorch-ext/*/_ops.py\n_pilot_test.py\n__pycache__/\n' > .gitignore
    [ -d .git ] || git init -q
    git add -A
    git -c user.email=build@local -c user.name=build commit -qm build 2>/dev/null || true
    nix run .#build-and-copy -L 2>&1 | tail -3
    echo "  -> $(ls build/ 2>/dev/null | wc -l) variants in build/"
  ) || echo "!!! $name FAILED"
done
