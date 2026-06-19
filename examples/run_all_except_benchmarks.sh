#!/usr/bin/env bash
# Run all examples except the benchmark (scaling_test.py).
#
# Each example is executed with the current Python interpreter, exactly as if
# invoked directly from the command line.
#
# Usage:
#   ./examples/run_all_except_benchmarks.sh

set -euo pipefail

EXAMPLES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"

# Examples to skip (e.g. the benchmark, which is slow and produces no artifact
# meant for the README gallery).
SKIP=("scaling_test.py")

failures=()
count=0

for script in "$EXAMPLES_DIR"/*.py; do
    name="$(basename "$script")"

    skip=false
    for s in "${SKIP[@]}"; do
        if [[ "$name" == "$s" ]]; then
            skip=true
            break
        fi
    done
    [[ "$skip" == true ]] && continue

    printf '\n%s\nRunning %s\n%s\n' "======================================================================" "$name" "======================================================================"
    count=$((count + 1))
    if ! "$PYTHON" "$script"; then
        failures+=("$name")
    fi
done

printf '\n%s\nSummary\n%s\n' "======================================================================" "======================================================================"
printf 'Ran %d example(s).\n' "$count"
if [[ ${#failures[@]} -gt 0 ]]; then
    printf 'Failed: %s\n' "${failures[*]}"
    exit 1
fi
printf 'All examples completed successfully.\n'
