# `part2/test_random_policy.py`

## Scopo

Questo script serve a prendere confidenza con `PandaPush-v3` senza fare training.

## Cosa fa

- crea l'ambiente
- stampa observation space e action space
- esegue episodi con azioni casuali
- opzionalmente renderizza la simulazione

## Perche' e' utile

E' il modo piu' rapido per verificare:

- che l'ambiente sia installato correttamente
- che il rendering funzioni
- che il formato delle osservazioni e delle azioni sia quello atteso

## Flusso

Per ogni episodio:

1. chiama `env.reset()`
2. campiona un'azione casuale con `env.action_space.sample()`
3. esegue `env.step(action)`
4. continua finche' l'episodio termina

## Quando usarlo

Conviene lanciarlo:

- prima di fare training
- dopo installazione di `panda-gym`
- se sospetti problemi di ambiente o rendering

## Esempio

```bash
python part2/test_random_policy.py
```

## Relazione con gli altri script

Questo script non usa:

- `RandomizationWrapper`
- `SAC`
- `PPO`

Serve solo come controllo iniziale e strumento di debugging dell'ambiente.
