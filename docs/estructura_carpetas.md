# Estructura de carpetas del proyecto `task-assignment-robotic-warehouse`

Este documento resume la estructura actual del repositorio, con foco en los modulos `tarware` (base del entorno) y `tarware_ext` (extensiones para grafos, politicas y entrenamiento).

## 1) Vista general del repositorio

```text
task-assignment-robotic-warehouse/
├── demos/                  # Demos y ejemplos de ejecucion
├── docs/                   # Documentacion y recursos
├── eval/                   # Resultados de evaluacion por entorno/politica
├── scripts/                # Scripts CLI para demo, entrenamiento y evaluacion
├── tarware/                # Modulo base del entorno de warehouse
├── tarware_ext/            # Modulo extendido (grafos, politicas, runners, training)
├── tests/                  # Pruebas unitarias e integracion
├── pyproject.toml          # Configuracion del proyecto/paquetes
├── README.md               # Guia principal del repositorio
└── *.csv / logs varios     # Artefactos de evaluaciones y ejecuciones
```

## 2) Modulo `tarware` (core del entorno)

Ruta: `tarware/`

```text
tarware/
├── __init__.py
├── definitions.py          # Constantes, enums y definiciones comunes del entorno
├── heuristic.py            # Heuristicas base de asignacion/comportamiento
├── rendering.py            # Renderizado/visualizacion del entorno
├── smoke_tarware.py        # Smoke test/utilidad rapida para validar ejecucion
├── test.py                 # Pruebas basicas del modulo
├── warehouse.py            # Logica principal del simulador warehouse
├── spaces/                 # Espacios de observacion multiagente
│   ├── MultiAgentBaseObservationSpace.py
│   ├── MultiAgentGlobalObservationSpace.py
│   └── MultiAgentPartialObservationSpace.py
└── utils/                  # Utilidades de apoyo
    ├── utils.py
    └── wrappers.py
```

### Rol de `tarware`

- Implementa el entorno base de simulacion del almacen.
- Define espacios de observacion para distintos niveles de informacion.
- Proporciona componentes reutilizables para heuristicas y visualizacion.

## 3) Modulo `tarware_ext` (extensiones sobre el core)

Ruta: `tarware_ext/`

```text
tarware_ext/
├── __init__.py
├── envs/                   # Registro/adaptadores de entornos
│   ├── registry.py
│   └── tarware_adapter.py
├── graphs/                 # Construccion y serializacion de representaciones en grafo
│   ├── builder.py
│   ├── builder_v0.py
│   ├── features.py
│   ├── masks.py
│   ├── schema.py
│   ├── serializer.py
│   └── utils.py
├── logs/                   # Logging estructurado para experimentos
│   ├── csv_logger.py
│   └── jsonl_logger.py
├── policies/               # Politicas de decision (heuristica, greedy, score, GNN)
│   ├── base.py
│   ├── heuristic_policy.py
│   ├── random_policy.py
│   ├── graph_greedy_policy.py
│   ├── graph_score_policy.py
│   ├── gnn_policy.py
│   └── torch_gnn_policy.py
├── runners/                # Ejecucion de rollouts y evaluaciones
│   ├── evaluate.py
│   ├── metrics.py
│   ├── rollout.py
│   └── seeds.py
└── training/               # Algoritmos y utilidades de entrenamiento RL
    ├── algo_mappo.py
    ├── algo_ppo.py
    ├── buffer.py
    └── utils.py
```

### Rol de `tarware_ext`

- Agrega una capa experimental para politicas avanzadas y pipelines de entrenamiento.
- Convierte el estado del entorno a estructuras de grafo para modelos GNN.
- Centraliza evaluacion, metricas y logging de experimentos.

## 4) Relacion entre `tarware` y `tarware_ext`

- `tarware` define el entorno y la dinamica base.
- `tarware_ext` consume/adapta ese entorno para:
  - construir estados en grafo,
  - aplicar politicas (incluyendo GNN),
  - entrenar y evaluar agentes,
  - registrar resultados reproducibles.

En conjunto, `tarware` aporta el simulador y `tarware_ext` la capa de investigacion/ML sobre ese simulador.
