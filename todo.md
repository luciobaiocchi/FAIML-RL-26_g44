# RL Experiment Todo

## Prima di lanciare gli esperimenti

- [ ] Sistemare `part2/train_sb3_ppo.py`: `ent_coef` deve essere numerico, ad esempio `0.01`
- [ ] Controllare che i nomi dei modelli salvati siano coerenti: `ppo_push_...` e `sac_push_...`
- [ ] Implementare `part2/rand_wrapper.py` per supportare davvero `none`, `udr` e `adr`
- [ ] Sistemare `part2/eval_sb3.py` perché ora supporta solo `SAC`
- [ ] Verificare che `panda-gym` sia installato con:

```bash
cd part2/panda-gym
pip install -e .
```

## Budget consigliato

- [ ] Debug veloce: `100000` step
- [ ] Run finale: `500000` step
- [ ] Evaluation rapida: `100` episodi
- [ ] Evaluation finale: `500` episodi

## Significato degli esperimenti

- [ ] `source -> source` = baseline
- [ ] `source -> target` = lower bound di transfer
- [ ] `target -> target` = upper bound / oracle

## Piano completo degli esperimenti

### 1. Esperimenti con `none`

- [ ] Train `source` con `none`, poi eval su `source`
- [ ] Train `source` con `none`, poi eval su `target`
- [ ] Train `target` con `none`, poi eval su `target`

Comandi:

```bash
python part2/train_sb3_sac.py --sampling-strategy none --env-type source --timesteps 500000
python part2/train_sb3_sac.py --sampling-strategy none --env-type target --timesteps 500000

python part2/train_sb3_ppo.py --sampling-strategy none --env-type source --timesteps 500000
python part2/train_sb3_ppo.py --sampling-strategy none --env-type target --timesteps 500000
```

### 2. Esperimenti con `udr`

- [ ] Train `source` con `udr`, poi eval su `source`
- [ ] Train `source` con `udr`, poi eval su `target`
- [ ] Train `target` con `udr`, poi eval su `target`

Comandi:

```bash
python part2/train_sb3_sac.py --sampling-strategy udr --env-type source --timesteps 500000
python part2/train_sb3_sac.py --sampling-strategy udr --env-type target --timesteps 500000

python part2/train_sb3_ppo.py --sampling-strategy udr --env-type source --timesteps 500000
python part2/train_sb3_ppo.py --sampling-strategy udr --env-type target --timesteps 500000
```

### 3. Esperimenti con `adr`

- [ ] Train `source` con `adr`, poi eval su `source`
- [ ] Train `source` con `adr`, poi eval su `target`
- [ ] Train `target` con `adr`, poi eval su `target`

Comandi:

```bash
python part2/train_sb3_sac.py --sampling-strategy adr --env-type source --timesteps 500000
python part2/train_sb3_sac.py --sampling-strategy adr --env-type target --timesteps 500000

python part2/train_sb3_ppo.py --sampling-strategy adr --env-type source --timesteps 500000
python part2/train_sb3_ppo.py --sampling-strategy adr --env-type target --timesteps 500000
```

## Griglia minima consigliata se il tempo è poco

- [ ] `none`: fare `source -> source`, `source -> target`, `target -> target`
- [ ] `udr`: train su `source`, eval su `source` e `target`
- [ ] `adr`: train su `source`, eval su `source` e `target`

Questa versione riduce i run mantenendo il confronto più importante:

- [ ] baseline in-domain
- [ ] trasferimento senza randomization
- [ ] trasferimento con `udr`
- [ ] trasferimento con `adr`
- [ ] upper bound addestrando direttamente su `target`

## Ordine consigliato per risparmiare tempo

- [ ] Fare prima tutti i run brevi da `100000` step
- [ ] Controllare se il training migliora davvero
- [ ] Lanciare i run finali da `500000` step solo per i casi che funzionano
- [ ] Fare evaluation finale solo sui modelli buoni

Ordine pratico:

- [ ] `SAC none source 100k`
- [ ] `PPO none source 100k`
- [ ] eval dei due modelli su `source` e `target`
- [ ] scegliere l'algoritmo più promettente
- [ ] lanciare i run finali `500k`
- [ ] aggiungere poi `udr`
- [ ] aggiungere infine `adr`

## Cose da salvare per la relazione

- [ ] modello
- [ ] sampling strategy
- [ ] dominio di training
- [ ] dominio di evaluation
- [ ] numero di timesteps
- [ ] mean return
- [ ] std return
- [ ] success rate
- [ ] eventuali note su stabilità o fallimenti
