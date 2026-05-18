# Part 2 Docs

Questa cartella contiene la documentazione degli script Python principali usati in `part2`.

## File documentati

- [train_sb3.md](/Users/leonardo/Polito/FAIMDL/FAIML-RL-26_g44/docs/train_sb3.md): template incompleto per training SB3
- [train_sb3_sac.md](/Users/leonardo/Polito/FAIMDL/FAIML-RL-26_g44/docs/train_sb3_sac.md): training con SAC
- [train_sb3_ppo.md](/Users/leonardo/Polito/FAIMDL/FAIML-RL-26_g44/docs/train_sb3_ppo.md): training con PPO
- [eval_sb3.md](/Users/leonardo/Polito/FAIMDL/FAIML-RL-26_g44/docs/eval_sb3.md): evaluation dei modelli salvati
- [rand_wrapper.md](/Users/leonardo/Polito/FAIMDL/FAIML-RL-26_g44/docs/rand_wrapper.md): domain randomization wrapper per massa oggetto
- [test_random_policy.md](/Users/leonardo/Polito/FAIMDL/FAIML-RL-26_g44/docs/test_random_policy.md): script base per esplorare l'ambiente

## Flusso consigliato

1. Verificare che `panda-gym` sia installato in editable mode.
2. Testare l'ambiente con `test_random_policy.py`.
3. Addestrare un modello con `train_sb3_sac.py` o `train_sb3_ppo.py`.
4. Usare `rand_wrapper.py` per i casi `none`, `udr` e `adr`.
5. Valutare i modelli con `eval_sb3.py`.

## Nota importante

Al momento:

- `train_sb3_sac.py` e `train_sb3_ppo.py` sono collegati al `RandomizationWrapper`
- `rand_wrapper.py` e' stato implementato
- `eval_sb3.py` supporta ancora solo `SAC`
- `train_sb3.py` e' ancora un template incompleto
