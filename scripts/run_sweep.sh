#!/usr/bin/env bash
# scripts/run_sweep.sh

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$DIR/run_experiments.sh" --mode sweep "$@"
