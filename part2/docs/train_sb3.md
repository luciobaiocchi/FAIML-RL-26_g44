# `part2/train_sb3.py`

## Scopo

Questo file sembra essere un template generico per training con Stable-Baselines3, probabilmente pensato come base prima di separare `SAC` e `PPO`.

## Stato attuale

Al momento e' incompleto.

Ha:

- parser degli argomenti
- creazione dell'ambiente
- import di `DDPG`
- import di `RandomizationWrapper`

Ma non ha:

- applicazione del wrapper
- creazione del modello
- training
- salvataggio del modello

## Osservazioni importanti

- la descrizione dell'argparser dice `"Train SAC on PandaPush-v3"` ma il file importa `DDPG`
- il nome del file salvato usa ancora prefisso `sac_push_...`
- ci sono import non usati come `deque` e `numpy`

Quindi il file va visto come un residuo/template, non come script pronto.

## Cosa fare se vuoi usarlo

Hai due strade:

1. lasciarlo come template e usi solo `train_sb3_sac.py` e `train_sb3_ppo.py`
2. lo completi davvero per `DDPG`

Se il progetto richiede solo SAC e PPO, la scelta piu' semplice e' non usarlo.
