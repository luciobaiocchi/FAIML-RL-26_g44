# `part2/train_sb3_ppo.py`

## Scopo

Questo script addestra un agente `PPO` su `PandaPush-v3` usando Stable-Baselines3.

## Argomenti supportati

- `--sampling-strategy`: `none`, `udr`, `adr`
- `--env-type`: `source`, `target`
- `--timesteps`: numero totale di passi di training

## Flusso dello script

1. Legge gli argomenti da riga di comando.
2. Crea l'ambiente `PandaPush-v3`.
3. Applica `RandomizationWrapper`.
4. Crea il modello `PPO`.
5. Avvia il training con `learn(...)`.
6. Salva il modello finale.

## Modifiche introdotte

Questo file aveva due problemi principali:

1. il wrapper veniva importato ma non usato
2. `ent_coef` era stato inizialmente impostato con una sintassi stile SAC

Le modifiche fatte sono:

- applicazione del wrapper anche a PPO
- uso di:

```python
ent_coef=0.01
```

invece di una stringa del tipo `"auto_0.1"`

- naming coerente del file salvato:

```python
ppo_push_{sampling_strategy}_{env_type}_{timesteps // 1000}k
```

## Perche' `ent_coef` e' diverso da SAC

In PPO:

- `ent_coef` deve essere un numero
- controlla quanto incentivare l'esplorazione

In SAC:

- `ent_coef` puo' essere una stringa come `"auto_0.1"`
- in quel caso viene adattato automaticamente

Questa differenza era la causa del bug iniziale in PPO.

## Parametri principali del modello

- policy: `MultiInputPolicy`
- algoritmo: `PPO`
- `ent_coef=0.01`
- `verbose=2`

## Esempi d'uso

```bash
python part2/train_sb3_ppo.py --sampling-strategy none --env-type source --timesteps 100000
python part2/train_sb3_ppo.py --sampling-strategy udr --env-type source --timesteps 500000
python part2/train_sb3_ppo.py --sampling-strategy adr --env-type source --timesteps 500000
```

## Osservazioni

- PPO e SAC non hanno gli stessi iperparametri
- i risultati possono essere piu' sensibili alla lunghezza del training
- conviene prima fare run corti da debug e poi i run finali
