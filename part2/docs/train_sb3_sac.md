# `part2/train_sb3_sac.py`

## Scopo

Questo script addestra un agente `SAC` su `PandaPush-v3` usando Stable-Baselines3, con supporto opzionale a **HER (Hindsight Experience Replay)**.

## Argomenti supportati

| Argomento | Valori | Default | Descrizione |
|---|---|---|---|
| `--sampling-strategy` | `none`, `udr`, `adr` | `none` | Strategia di randomizzazione della massa |
| `--env-type` | `source`, `target` | `source` | Tipo di ambiente |
| `--timesteps` | intero | `500000` | Passi totali di training |
| `--seed` | intero | `42` | Seed per riproducibilità |
| `--save` | `True`, `False` | `False` | Salva i pesi del modello |
| `--her` | flag | disabilitato | Abilita Hindsight Experience Replay |

## Flusso dello script

1. Legge gli argomenti da riga di comando.
2. Crea l'ambiente `PandaPush-v3`.
3. Applica `RandomizationWrapper`.
4. Crea il modello `SAC` con policy `MultiInputPolicy` (con o senza HER).
5. Lancia `model.learn(...)`.
6. Salva il modello in un file `.zip`.

## Parametri principali del modello

- policy: `MultiInputPolicy`
- algoritmo: `SAC`
- `ent_coef="auto_0.1"`: coefficiente entropico appreso automaticamente con valore iniziale 0.1
- `verbose=1`
- `tensorboard_log="./sac_logs/"`: log per TensorBoard

## HER — Hindsight Experience Replay

### Cos'è

HER è una tecnica di data augmentation per il replay buffer, progettata specificamente per task **goal-conditioned** come PandaPush. È stata introdotta da Andrychowicz et al. (2017) e funziona particolarmente bene con SAC.

### Il problema che risolve

In un task come PandaPush, il reward è quasi sempre negativo (l'agente non riesce a spingere il cubo esattamente sul goal). Questo significa che il replay buffer è pieno di episodi falliti con reward ≈ -1, e il segnale di apprendimento è debolissimo: la rete non vede quasi mai cosa succede quando si riesce.

### Come funziona

Alla fine di ogni episodio, l'agente ha mosso il cubo in una posizione `achieved_goal` diversa dal `desired_goal` originale. HER si chiede: *"e se il goal fosse stato proprio `achieved_goal`?"*

Prende le transizioni dell'episodio fallito, sostituisce `desired_goal` con `achieved_goal`, e ricalcola il reward — che ora è positivo perché il goal è stato raggiunto. Queste transizioni rilabellate vengono aggiunte al replay buffer insieme a quelle reali.

```
Episodio reale:    (s, a, achieved_goal=X, desired_goal=G) → reward=-1  (fallito)
Transizione HER:   (s, a, achieved_goal=X, desired_goal=X) → reward=+1  (successo!)
```

PandaPush ha esattamente `achieved_goal` e `desired_goal` nell'observation dict, quindi HER funziona nativamente senza modifiche all'ambiente.

### Parametri usati

```python
HerReplayBuffer(
    n_sampled_goal=4,               # per ogni transizione reale, aggiunge 4 transizioni HER
    goal_selection_strategy="future" # i goal alternativi sono scelti da step futuri dello stesso episodio
)
```

`goal_selection_strategy="future"` è la strategia più efficace in pratica: per ogni transizione al tempo `t`, campiona goal da step `t+1, t+2, ...` dello stesso episodio. Questo garantisce che i goal scelti siano sempre raggiungibili dato lo stato corrente.

### Perché SAC + HER funziona meglio di PPO

SAC è off-policy: usa un replay buffer e riutilizza le transizioni passate più volte. HER si integra direttamente in questo buffer, moltiplicando l'utilità di ogni episodio. PPO invece è on-policy e butta i dati dopo ogni update, rendendo HER inapplicabile.

## Nome del file salvato

```
sac_push_{sampling_strategy}_{env_type}_{timesteps // 1000}k[_her].zip
```

Esempi:

```
sac_push_none_source_500k.zip
sac_push_udr_source_500k_her.zip
```

## Esempi d'uso

```bash
# SAC base
python train_sb3_sac.py --sampling-strategy none --env-type source --timesteps 100000

# SAC + HER (consigliato per massimizzare success rate)
python train_sb3_sac.py --sampling-strategy none --env-type source --timesteps 100000 --her

# SAC + HER + UDR + salvataggio
python train_sb3_sac.py --sampling-strategy udr --env-type source --timesteps 500000 --her --save True
```

## Osservazioni

- `source` usa la dinamica con massa nominale leggera, `target` con massa pesante
- con `udr` e `adr`, la randomizzazione avviene indipendentemente dall'`env-type`
- i log TensorBoard vengono salvati in `./sac_logs/`; per visualizzarli: `tensorboard --logdir ./sac_logs/`
