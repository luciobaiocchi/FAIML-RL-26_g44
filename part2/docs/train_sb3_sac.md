# `part2/train_sb3_sac.py`

## Scopo

Questo script addestra un agente `SAC` su `PandaPush-v3` usando Stable-Baselines3.

## Argomenti supportati

- `--sampling-strategy`: `none`, `udr`, `adr`
- `--env-type`: `source`, `target`
- `--timesteps`: numero totale di passi di training

## Flusso dello script

1. Legge gli argomenti da riga di comando.
2. Crea l'ambiente `PandaPush-v3`.
3. Applica `RandomizationWrapper`.
4. Crea il modello `SAC` con policy `MultiInputPolicy`.
5. Lancia `model.learn(...)`.
6. Salva il modello in un file `.zip`.

## Modifiche introdotte

La versione iniziale del file:

- importava `RandomizationWrapper` ma non lo usava
- non applicava davvero la randomizzazione

Le modifiche fatte sono:

- applicazione esplicita del wrapper:

```python
env = RandomizationWrapper(
    env,
    mass_range=(1.0, 5.0),
    mode=args.sampling_strategy,
)
```

- pulizia della creazione del modello `SAC`
- mantenimento del salvataggio finale coerente con la strategia selezionata

## Parametri principali del modello

- policy: `MultiInputPolicy`
- algoritmo: `SAC`
- `ent_coef="auto_0.1"`
- `verbose=2`

`ent_coef="auto_0.1"` e' corretto per SAC perche' abilita un coefficiente entropico appreso automaticamente con inizializzazione controllata.

## Nome del file salvato

Il modello viene salvato con:

```python
sac_push_{sampling_strategy}_{env_type}_{timesteps // 1000}k
```

Esempio:

```bash
sac_push_udr_source_500k.zip
```

## Esempi d'uso

```bash
python part2/train_sb3_sac.py --sampling-strategy none --env-type source --timesteps 100000
python part2/train_sb3_sac.py --sampling-strategy udr --env-type source --timesteps 500000
python part2/train_sb3_sac.py --sampling-strategy adr --env-type source --timesteps 500000
```

## Osservazioni

- `source` usa la dinamica con massa nominale leggera
- `target` usa la dinamica con massa nominale pesante
- con `udr` e `adr`, la randomizzazione avviene anche se `env-type` e' `source` o `target`; quello che cambia e' la massa nominale iniziale del dominio
