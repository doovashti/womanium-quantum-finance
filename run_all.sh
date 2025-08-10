#!/usr/bin/env bash
set -e

# optional: ensure we're in the repo root
cd "$(dirname "$0")"

echo "[1/3] Build demo + QUBO quick smoke"
python -m src.problem_demo >/dev/null
python -m src.qubo >/dev/null

echo "[2/3] Run VQE"
python -m src.vqe_solve

echo "[3/3] Compare against exact + SA"
python -m src.compare

echo "Done. Artifacts in ./results:"
ls -1 results
