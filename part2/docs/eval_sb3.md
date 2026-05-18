# `part2/eval_sb3.py`

## Scopo

Questo script valuta un modello salvato su `PandaPush-v3` e stampa statistiche aggregate sulle performance.

## Cosa fa

Per ogni episodio:

1. resetta l'ambiente
2. lascia che il modello predica le azioni
3. accumula il reward totale
4. registra il successo finale se `info["is_success"]` e' presente

Alla fine stampa:

- numero episodi
- return medio
- deviazione standard
- return minimo
- return massimo
- success rate

## Argomenti supportati

- `--model-path`: path al file `.zip`
- `--episodes`: numero di episodi di evaluation
- `--stochastic`: usa la policy in modo non deterministico
- `--render`: apre il rendering umano
- `--env-type`: `source` oppure `target`

## Limite attuale importante

Lo script carica il modello usando:

```python
model = SAC.load(model_path, env=env)
```

Questo significa che attualmente:

- funziona per modelli `SAC`
- non funziona direttamente per modelli `PPO`

Se vuoi valutare anche PPO, lo script va esteso con una selezione dell'algoritmo o un loader automatico.

## Ruolo negli esperimenti

Questo script e' quello che usi per confronti come:

- `source -> source`
- `source -> target`
- `target -> target`

Per esempio:

- alleni `sac_push_none_source_500k.zip`
- lo valuti su `source` per la baseline
- lo valuti su `target` per il lower bound di transfer

## Esempi d'uso

```bash
python part2/eval_sb3.py --model-path sac_push_none_source_500k.zip --env-type source --episodes 100
python part2/eval_sb3.py --model-path sac_push_none_source_500k.zip --env-type target --episodes 100
python part2/eval_sb3.py --model-path sac_push_none_target_500k.zip --env-type target --episodes 100
```

## Nota sulla determinicita'

Per risultati confrontabili conviene usare il default deterministico.

L'opzione `--stochastic` e' utile soprattutto per debug o per capire quanto la policy dipenda dalla parte esplorativa.
