# Part 2 Docs

Questa cartella contiene la documentazione degli script Python principali usati in `part2`.

## File documentati

- [train_sb3.md](train_sb3.md): script unificato per training con PPO o SAC
- [train_sb3_sac.md](train_sb3_sac.md): training con SAC (standalone, con HER)
- [train_sb3_ppo.md](train_sb3_ppo.md): training con PPO (standalone, multi-env)
- [run_tests.md](run_tests.md): grid search su parametri divisa in 4 profili
- [eval_sb3.md](eval_sb3.md): evaluation dei modelli salvati
- [rand_wrapper.md](rand_wrapper.md): domain randomization wrapper per massa oggetto
- [wrappers.md](wrappers.md): reward shaping wrapper
- [test_random_policy.md](test_random_policy.md): script base per esplorare l'ambiente
- [research_directions.md](research_directions.md): survey letteratura e direzioni di ricerca per paper

## Flusso consigliato

1. Verificare che `panda-gym` sia installato in editable mode.
2. Testare l'ambiente con `test_random_policy.py`.
3. Addestrare un modello con `train_sb3.py --algorithm sac|ppo`.
4. Fare grid search con `run_tests.py --algorithm sac|ppo --profile 1-4`.
5. Valutare i modelli con `eval_sb3.py`.

## Note

- `train_sb3.py` è lo script unificato che sostituisce `train_sb3_sac.py` e `train_sb3_ppo.py`
- `run_tests.py` usa `train_sb3.py` internamente — assicurarsi che sia presente
- Attivare il proprio env conda prima di lanciare qualsiasi script
