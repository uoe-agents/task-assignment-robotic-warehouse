#!/usr/bin/env bash
set -u

# Run demos for the same env/policy/top-k combinations used by run_evals.sh
PYTHON=${PYTHON:-.venv/bin/python}
TOPKS=(2 4)
ENVS=(
  tarware-medium-4agvs-2pickers-globalobs-v1
  tarware-large-8agvs-4pickers-globalobs-v1
)
POLICIES=(graph_score graph_greedy gnn torch_gnn)

mkdir -p demos

echo "Using PYTHON=${PYTHON}"

for env_id in "${ENVS[@]}"; do
  if [[ "${env_id}" == *"medium"* ]]; then
    STEPS=${STEPS_MEDIUM:-400}
  else
    STEPS=${STEPS_LARGE:-500}
  fi

  for policy in "${POLICIES[@]}"; do
    case "${policy}" in
      graph_greedy)
        out="demos/${env_id}_${policy}.gif"
        echo "RUN DEMO: env=${env_id} policy=${policy} steps=${STEPS} -> ${out}"
        "${PYTHON}" scripts/demo_visualize.py --env-id "${env_id}" --policy "${policy}" --steps "${STEPS}" --save-gif "${out}" || echo "FAILED: ${env_id} ${policy}" >> demos/errors.log
        ;;
      torch_gnn)
        if "${PYTHON}" -c "import torch" >/dev/null 2>&1; then
          for k in "${TOPKS[@]}"; do
            out="demos/${env_id}_${policy}_top${k}.gif"
            echo "RUN DEMO: env=${env_id} policy=${policy} top_k=${k} steps=${STEPS} -> ${out}"
            "${PYTHON}" scripts/demo_visualize.py --env-id "${env_id}" --policy "${policy}" --top-k "${k}" --steps "${STEPS}" --save-gif "${out}" || echo "FAILED: ${env_id} ${policy} top ${k}" >> demos/errors.log
          done
        else
          echo "Skipping ${policy} for ${env_id} (torch not available)" | tee -a demos/errors.log
        fi
        ;;
      *)
        for k in "${TOPKS[@]}"; do
          out="demos/${env_id}_${policy}_top${k}.gif"
          echo "RUN DEMO: env=${env_id} policy=${policy} top_k=${k} steps=${STEPS} -> ${out}"
          "${PYTHON}" scripts/demo_visualize.py --env-id "${env_id}" --policy "${policy}" --top-k "${k}" --steps "${STEPS}" --save-gif "${out}" || echo "FAILED: ${env_id} ${policy} top ${k}" >> demos/errors.log
        done
        ;;
    esac
  done
done

echo "Done. GIFs are in demos/; CSVs are in eval/"