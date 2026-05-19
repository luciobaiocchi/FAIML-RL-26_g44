# `part2/run_tests.py`

## Scopo

Esegue una grid search su un sottoinsieme di parametri per SAC o PPO, organizzata in 4 profili predefiniti. Ogni combinazione di parametri viene eseguita come processo separato e i log vengono salvati in `test/profile_N/<run_name>/training_log.txt`.

## Prerequisiti

Attivare il proprio ambiente conda prima di lanciare lo script:

```bash
conda activate <nome_env>
python run_tests.py --algorithm sac --profile 1
```

Lo script usa `sys.executable` (il Python dell'env attivo) per lanciare i sottoprocessi — non gestisce l'attivazione dell'env autonomamente.

## Argomenti

| Argomento | Valori | Obbligatorio | Descrizione |
|---|---|---|---|
| `--algorithm` | `sac`, `ppo` | Sì | Algoritmo da usare |
| `--profile` | `1`, `2`, `3`, `4` | Sì | Profilo di parametri da eseguire |
| `--smoke` | flag | No | Esegue ogni run con pochi timesteps per verificare che non crashino |

## Profili SAC

| Profilo | Parametri esplorati | Run totali |
|---|---|---|
| 1 | `learning_rate` × `batch_size` — baseline senza HER e DR | 4 |
| 2 | `learning_rate` × `gradient_steps` — impatto degli update per step | 6 |
| 3 | `learning_rate` × `her` × `sampling_strategy` — impatto di HER e DR | 8 |
| 4 | `batch_size` × `gradient_steps` × `her=True` × `sampling_strategy` — combo avanzata | 8 |

## Profili PPO

| Profilo | Parametri esplorati | Run totali |
|---|---|---|
| 1 | `n_envs` — baseline senza DR | 2 |
| 2 | `n_envs` × `sampling_strategy` — parallelismo + DR | 4 |
| 3 | `n_envs` alti × `sampling_strategy` — scalabilità | 4 |
| 4 | `n_envs` × `sampling_strategy` — full grid | 9 |

## Struttura output

```
test/
└── profile_1/
    ├── sac_learning_rate3e-04_batch_size256_her0_sampling_strategynone/
    │   └── training_log.txt
    ├── sac_learning_rate3e-04_batch_size512_her0_sampling_strategynone/
    │   └── training_log.txt
    └── ...
```

Ogni `training_log.txt` contiene:
- Il comando eseguito
- Il timestamp di inizio
- L'output completo di stdout+stderr del processo

## Smoke test

Il flag `--smoke` riduce drasticamente i timesteps per verificare che ogni combinazione di parametri non produca errori:

| Algoritmo | Timesteps smoke |
|---|---|
| SAC | 500 (supera `learning_starts=100`, fa qualche update) |
| PPO | 3000 (circa 1 rollout con `n_steps=2048`) |

```bash
# Verifica veloce che tutto funzioni
python run_tests.py --algorithm sac --profile 3 --smoke
python run_tests.py --algorithm ppo --profile 2 --smoke
```

## Esempi d'uso

```bash
# Smoke test su tutti i profili SAC
python run_tests.py --algorithm sac --profile 1 --smoke
python run_tests.py --algorithm sac --profile 2 --smoke
python run_tests.py --algorithm sac --profile 3 --smoke
python run_tests.py --algorithm sac --profile 4 --smoke

# Run completo profilo 3 SAC (HER + DR)
python run_tests.py --algorithm sac --profile 3

# Run completo profilo 4 PPO (full grid)
python run_tests.py --algorithm ppo --profile 4
```

## Riepilogo finale

Al termine, lo script stampa un riepilogo con esito di ogni run:

```
============================================================
RIEPILOGO — 8 run
============================================================
  ✓  sac_learning_rate3e-04_her0_sampling_strategynone
  ✓  sac_learning_rate1e-03_her0_sampling_strategynone
  ✗  sac_learning_rate3e-04_her1_sampling_strategyudr
  ...

6/8 completate con successo.
```

Le run fallite hanno il log completo in `training_log.txt` per diagnosticare l'errore.

## Note

- Le run vengono eseguite sequenzialmente, non in parallelo — per parallelizzarle manualmente si può lanciare `run_tests.py` con profili diversi su terminali diversi
- Lo script chiama `train_sb3.py` (script unificato) — assicurarsi che sia presente nella stessa directory
- I log tensorboard vengono salvati nelle rispettive cartelle (`./ppo_logs/`, `./sac_logs/`) indipendentemente dal profilo
