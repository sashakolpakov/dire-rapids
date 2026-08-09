#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(git rev-parse --show-toplevel)"
HARNESS="$REPOSITORY_ROOT/benchmarking/bench_topology_historical.py"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_ROOT="${1:-$REPOSITORY_ROOT/issue14-historical-results}"

if [[ "$OUTPUT_ROOT" != /* ]]; then
  OUTPUT_ROOT="$REPOSITORY_ROOT/$OUTPUT_ROOT"
fi
if [[ "$OUTPUT_ROOT" == "/" ]]; then
  echo "Refusing to use / as the result root" >&2
  exit 2
fi

WORK_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/dire-issue14.XXXXXX")"
WORKTREES=()

cleanup() {
  local worktree
  for worktree in "${WORKTREES[@]}"; do
    if [[ -e "$worktree/.git" ]]; then
      git -C "$REPOSITORY_ROOT" worktree remove --force "$worktree" >/dev/null
    fi
  done
  rmdir "$WORK_ROOT" 2>/dev/null || true
}
trap cleanup EXIT

add_worktree() {
  local name="$1"
  local revision="$2"
  local destination="$WORK_ROOT/$name"
  git -C "$REPOSITORY_ROOT" cat-file -e "$revision^{commit}"
  git -C "$REPOSITORY_ROOT" worktree add --detach "$destination" "$revision"
  WORKTREES+=("$destination")
}

mkdir -p "$OUTPUT_ROOT/raw" "$OUTPUT_ROOT/summary"

if [[ -f "$OUTPUT_ROOT/frozen-datasets/manifest.json" ]]; then
  "$PYTHON_BIN" "$HARNESS" verify-datasets \
    --root "$OUTPUT_ROOT/frozen-datasets"
else
  "$PYTHON_BIN" "$HARNESS" prepare \
    --output "$OUTPUT_ROOT/frozen-datasets"
fi

add_worktree preset-introduction 9117dc45a3e130fa1d636dfd181f3e97960c5b3b
add_worktree audited-main 293b622cc79fa8ea6fd5b54009e0930e3385b22f
add_worktree pr12-head 471ac168eb2e6638a84f700fe077f29f20e24488

"$PYTHON_BIN" "$HARNESS" run \
  --variant 9117dc4_index_search_flat \
  --source-root "$WORK_ROOT/preset-introduction" \
  --evaluator-source "$WORK_ROOT/audited-main/dire_rapids/betti_curve.py" \
  --datasets-root "$OUTPUT_ROOT/frozen-datasets" \
  --reference-cache "$OUTPUT_ROOT/reference-cache" \
  --output "$OUTPUT_ROOT/raw/9117dc4_index_search_flat.jsonl"

"$PYTHON_BIN" "$HARNESS" run \
  --variant 293b622_index_search_flat \
  --source-root "$WORK_ROOT/audited-main" \
  --evaluator-source "$WORK_ROOT/audited-main/dire_rapids/betti_curve.py" \
  --datasets-root "$OUTPUT_ROOT/frozen-datasets" \
  --reference-cache "$OUTPUT_ROOT/reference-cache" \
  --output "$OUTPUT_ROOT/raw/293b622_index_search_flat.jsonl"

"$PYTHON_BIN" "$HARNESS" run \
  --variant 471ac16_index_search_flat \
  --source-root "$WORK_ROOT/pr12-head" \
  --evaluator-source "$WORK_ROOT/audited-main/dire_rapids/betti_curve.py" \
  --datasets-root "$OUTPUT_ROOT/frozen-datasets" \
  --reference-cache "$OUTPUT_ROOT/reference-cache" \
  --output "$OUTPUT_ROOT/raw/471ac16_index_search_flat.jsonl"

"$PYTHON_BIN" "$HARNESS" run \
  --variant 471ac16_auto_all_neighbors \
  --source-root "$WORK_ROOT/pr12-head" \
  --evaluator-source "$WORK_ROOT/audited-main/dire_rapids/betti_curve.py" \
  --datasets-root "$OUTPUT_ROOT/frozen-datasets" \
  --reference-cache "$OUTPUT_ROOT/reference-cache" \
  --output "$OUTPUT_ROOT/raw/471ac16_auto_all_neighbors.jsonl"

"$PYTHON_BIN" "$HARNESS" summarize \
  --inputs "$OUTPUT_ROOT"/raw/*.jsonl \
  --output "$OUTPUT_ROOT/summary"

echo "Issue #14 historical audit completed: $OUTPUT_ROOT/summary"
