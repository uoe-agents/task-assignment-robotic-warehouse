#!/usr/bin/env bash
# Run evaluation sweeps for selected envs/policies and save per-episode CSVs
# Usage: bash scripts/run_evals.sh
# Optional env vars: PYTHON (default .venv/bin/python), EPISODES (default 10), STEPS_MEDIUM, STEPS_LARGE

set -u

PYTHON=${PYTHON:-.venv/bin/python}
EPISODES=${EPISODES:-10}
TOPKS=(2 4)
ENVS=(
  tarware-medium-4agvs-2pickers-globalobs-v1
  tarware-large-8agvs-4pickers-globalobs-v1
)
POLICIES=(graph_score graph_greedy gnn torch_gnn)

mkdir -p eval
mkdir -p demos

echo "Using PYTHON=${PYTHON}, EPISODES=${EPISODES}"

for env_id in "${ENVS[@]}"; do
  if [[ "${env_id}" == *"medium"* ]]; then
    STEPS=${STEPS_MEDIUM:-400}
  else
    STEPS=${STEPS_LARGE:-500}
  fi

  for policy in "${POLICIES[@]}"; do
    case "${policy}" in
      graph_greedy)
        out="eval/${env_id}_${policy}.csv"
        echo "RUN: env=${env_id} policy=${policy} episodes=${EPISODES} steps=${STEPS} -> ${out}"
        "${PYTHON}" scripts/eval.py --env-id "${env_id}" --policy "${policy}" --distance manhattan --episodes "${EPISODES}" --steps "${STEPS}" --csv "${out}" || echo "FAILED: ${env_id} ${policy}" >> eval/errors.log
        ;;
      torch_gnn)
        # Only attempt if torch is importable in the chosen Python
        if "${PYTHON}" -c "import torch" >/dev/null 2>&1; then
          for k in "${TOPKS[@]}"; do
            out="eval/${env_id}_${policy}_top${k}.csv"
            echo "RUN: env=${env_id} policy=${policy} top_k=${k} episodes=${EPISODES} steps=${STEPS} -> ${out}"
            "${PYTHON}" scripts/eval.py --env-id "${env_id}" --policy "${policy}" --distance manhattan --top-k "${k}" --episodes "${EPISODES}" --steps "${STEPS}" --csv "${out}" || echo "FAILED: ${env_id} ${policy} top ${k}" >> eval/errors.log
          done
        else
          echo "Skipping ${policy} for ${env_id} (torch not available)" | tee -a eval/errors.log
        fi
        ;;
      *)
        # graph_score, gnn: iterate top-k values
        for k in "${TOPKS[@]}"; do
          out="eval/${env_id}_${policy}_top${k}.csv"
          echo "RUN: env=${env_id} policy=${policy} top_k=${k} episodes=${EPISODES} steps=${STEPS} -> ${out}"
          "${PYTHON}" scripts/eval.py --env-id "${env_id}" --policy "${policy}" --distance manhattan --top-k "${k}" --episodes "${EPISODES}" --steps "${STEPS}" --csv "${out}" || echo "FAILED: ${env_id} ${policy} top ${k}" >> eval/errors.log
        done
        ;;
    esac
  done
done

echo "Done. CSV results are in the eval/ directory; visual demos go to demos/." 
