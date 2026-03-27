# Guia simple del flujo experimental

Esta guia explica, en pasos cortos, como corre un experimento con `tarware_ext` y `scripts/eval.py`.

## 1) Capas principales

- **scripts/**: puntos de entrada (CLI). Aqui eliges env, policy y parametros.
- **tarware_ext/**: capa experimental (runner, adapter, metrics, policies).
- **tarware/**: simulador base (env Gym + heuristica original).

## 2) Flujo basico (paso a paso)

1. `scripts/eval.py` crea el entorno con `gym.make(env_id)`.
2. Se selecciona la policy (random, heuristic, graph_greedy).
3. El runner ejecuta episodios y pasos.
4. El adapter normaliza `reset()` y `step()` en un `Transition` consistente.
   - Soporta `step()` con 4 o 5 valores (Gym vs Gymnasium).
5. La policy produce acciones (episodica o step-wise).
6. Metrics resume el episodio y el logger escribe el CSV.

## 3) Estado actual (verificado)

- **TarwareAdapter + Transition**: normaliza multi-agente, reward por agente/equipo, y done por agente/todos.
- **Runner (run_episode)**: soporta policies episodicas (`run_episode`) y step-wise (`act`).
- **evaluate**: corre N episodios, fija seeds por episodio y produce summary agregada.
- **metrics**: calcula deliveries, clashes, stucks, return, pick_rate y fps.
- **scripts/eval.py**: CLI unificada + export CSV por episodio.
- **Policies**:
  - `HeuristicPolicy`: episodica, delega en `tarware.heuristic.heuristic_episode`.
  - `GraphGreedyPolicy`: step-wise, usa `--distance` y limita AGVs con `--active-alpha` / `--max-active-agvs`.
  - `RandomPolicy`: sanity check.

## 4) Parametros clave

- `--env-id`: define el entorno (tamano, agentes, obs).
- `--policy`: random | heuristic | graph_greedy.
- `--episodes`, `--steps`, `--seed`: control del experimento.
- `--distance`: `manhattan` o `find_path` (solo graph_greedy).
- `--active-alpha`: limita AGVs activos. Regla base: `max_active_agvs = active_alpha * num_pickers`.
- `--max-active-agvs`: limite absoluto (si se pasa, sobreescribe la regla).
- `--csv` / `--no-csv`: salida de resultados.

## 5) Diagrama Mermaid (alto nivel)

```mermaid
flowchart TB
  subgraph S[scripts/]
    E[eval.py CLI\n--env-id --policy --episodes --steps --seed\n--distance --active-alpha --max-active-agvs --csv]
  end

  subgraph X[tarware_ext/]
    R[Runner / rollout.py\nrollout(env, policy)]
    A[TarwareAdapter\nreset()/step() -> Transition]
    M[Metrics / metrics.py\nsummarize_episode()]
    P1[HeuristicPolicy\n(episodic)]
    P2[GraphGreedyPolicy\n(step-wise)\ndistance_mode + active_alpha]
    T[Transition (normalized)\nobs\nreward_by_agent, reward_team\ndone_by_agent, done_all\ninfo]
  end

  subgraph C[tarware/ (core simulator)]
    ENV[Gym Env\nwarehouse.py + spaces/*\nreset()/step()]
    H[heuristic.py\n(baseline logic)]
  end

  E -->|gym.make(env_id)| ENV
  E -->|select policy| P1
  E -->|select policy| P2
  E -->|run episodes| R

  R --> A
  A -->|calls| ENV
  A -->|returns| T
  R -->|summarize| M
  M -->|writes via CSVLogger| CSV[(CSV file)]
  M -->|prints| OUT[Console summary]

  R -->|episodic path| P1
  P1 -->|delegates episode control| H

  R -->|step-wise path| P2
  P2 -->|act(env_unwrapped) -> actions| A

  E -.->|--distance manhattan/find_path| P2
  E -.->|--active-alpha / --max-active-agvs| P2
```

## 6) Ejemplo rapido

```bash
python scripts/eval.py \
  --env-id tarware-large-12agvs-7pickers-globalobs-v1 \
  --policy graph_greedy \
  --distance find_path \
  --active-alpha 3 \
  --episodes 5 \
  --steps 200 \
  --csv eval_graph_greedy_large_find.csv
```

## 7) Contrato GraphAssignmentEnv (SB3)

`GraphAssignmentEnv` expone un entorno Gym de agente unico para PPO, con una tarea
de alto nivel: asignacion explicita AGV -> request slot.

- **Action space**:
  - `MultiDiscrete([R+1] * num_agvs)`
  - `0` = no asignar ese AGV en este paso.
  - `1..R` = seleccionar slot del `request_queue` (indexado desde 1).
- **Observation space** (vector fijo):
  - Por AGV: `[y, x, busy, carrying, is_assigned, mission_type_code]`
  - Por cada slot de request (R):
    `[task_y, task_x, dist_this_agv, min_other_agv_dist, num_other_agvs_closer, valid_flag, is_assigned_flag]`
  - Global: `[num_tasks, num_free_agvs, num_busy_agvs, num_assigned_requests]`

Notas:

- `mission_type_code`: `0=no mission`, `1=PICKING`, `2=DELIVERING`, `3=RETURNING`.
- La alineacion accion-observacion se mantiene estricta por slot del `request_queue`.
- **Slots fijos durante el episodio**:
  - `R` se define en `GraphAssignmentConfig.max_request_slots`.
  - Si no se pasa, se infiere al crear el entorno desde `request_queue_size`
    (o longitud inicial de `request_queue` como fallback).

Nota de compatibilidad:

- `GraphAssignmentConfig.top_k` se mantiene solo como alias deprecado para
  `max_request_slots`.

- `GraphAssignmentConfig.obs_backend` permite elegir backend de observacion:
  - `assignment`: encoder directo desde env/controller (A).
  - `graph`: builder/proyeccion/encoder basados en `GraphState` (B).

## 8) Benchmark PPO vs Heuristica (assignment)

Para comparar `GraphAssignmentEnv` + PPO contra la heuristica baseline en
small/medium/large:

```bash
python scripts/benchmark_sb3_assignment.py \
  --env-ids tarware-small-2agvs-1pickers-globalobs-v1 tarware-medium-4agvs-2pickers-globalobs-v1 tarware-large-8agvs-4pickers-globalobs-v1 \
  --seeds 21 22 23 \
  --timesteps 20000 \
  --eval-episodes 10 \
  --steps 200 \
  --obs-backend assignment \
  --max-request-slots 20 \
  --csv eval/sb3_assignment_benchmark.csv
```

Notas de comparacion justa:

- Usa el mismo `--steps` para heuristica y PPO (el script ya los alinea).
- Usa multiples seeds y suficientes episodios para reducir varianza.
