#!/usr/bin/env bash
# scripts/run_many.sh

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$DIR/run_experiments.sh" --mode loop "$@"
