# Research Directions — Literature Survey

Survey condotto il 2026-05-19 su arxiv, Semantic Scholar, OpenReview.

---

## Background teorico

### Replay Buffer

Il replay buffer è la struttura dati centrale degli algoritmi off-policy come SAC. È una memoria circolare che accumula le transizioni `(stato, azione, reward, stato_successivo)` raccolte durante il training. Ad ogni step, SAC pesca un mini-batch casuale dal buffer per fare un gradient update, riutilizzando dati potenzialmente vecchi di ore.

```
step 1:  (s0, a0, r0, s1)  →  buffer: [(s0,a0,r0,s1)]
step 2:  (s1, a1, r1, s2)  →  buffer: [(s0,a0,r0,s1), (s1,a1,r1,s2)]
...
step N:  buffer pieno → sovrascrive la transizione più vecchia
```

Il vantaggio rispetto a PPO (on-policy): con `buffer_size=1M` e `batch_size=256`, ogni transizione viene usata in media ~4000 volte invece di una. Questo rende SAC molto più sample efficient.

### HER — Hindsight Experience Replay

HER è un meccanismo di data augmentation per il replay buffer, progettato per task goal-conditioned (come PandaPush) dove il reward è quasi sempre negativo — l'agente raramente raggiunge il goal esatto.

L'idea: quando un episodio fallisce e il cubo viene spostato in posizione `X` invece del goal `G`, HER si chiede *"e se il goal fosse stato X?"* — in quel caso l'episodio sarebbe stato un successo. Crea transizioni sintetiche rilabellando il goal:

```
Transizione reale:  (s, a, achieved=X, desired=G)  → reward=-1  (fallito)
Transizione HER:    (s, a, achieved=X, desired=X)  → reward= 0  (successo!)
```

Con `n_sampled_goal=4`, ogni transizione reale genera 4 transizioni HER aggiuntive usando goal futuri dello stesso episodio. Il buffer si riempie di esperienze positive molto più velocemente, risolvendo il problema del reward signal sparso.

### Perché il problema della coerenza fisica in HER+DR è non banale

Con domain randomization attiva, ogni episodio ha parametri fisici diversi (es. massa campionata). Le transizioni `(s, a, s')` nel buffer sono quindi generate da fisiche eterogenee. Quando HER rilabella una transizione, il reward viene ricalcolato tramite `env.compute_reward()` — per PandaPush questo è una semplice distanza dal goal, indipendente dalla massa.

Il problema non è nel reward, ma nel **Q-function**: esso vede la transizione `(s, a, s')` generata con `mass=2.3kg` e impara "questa azione porta al successo". Ma con `mass=4.5kg` la stessa azione produce uno `s'` completamente diverso. Se il buffer è dominato da episodi con massa bassa (più facili → più successi → più transizioni HER), la policy apprende principalmente a gestire oggetti leggeri e generalizza male al target domain con oggetti pesanti. Questa distorsione non è mai stata studiata.

### Come imparano i robot oggi e rilevanza di SAC+HER

Il campo del robot learning si è biforcato in due filoni principali.

**Imitation Learning** è la direzione dominante per manipulation (2023-2025). Invece di esplorare da zero con RL, si raccolgono dimostrazioni umane via teleoperazione e si fa behavioral cloning. Gli algoritmi più usati sono Diffusion Policy (Chi et al. 2023), che usa modelli diffusivi per generare azioni ed è stato lo stato dell'arte su manipulation tasks reali, ACT (Action Chunking with Transformers), usato su robot come ALOHA, e π0 di Physical Intelligence (2024), un foundation model pre-trainato su milioni di dimostrazioni. Questi approcci richiedono però un robot fisico e un sistema di teleoperazione — costi non banali per ricerca accademica.

**RL puro + sim-to-real** rimane il filone dominante per task dove le dimostrazioni sono difficili o costose da raccogliere. Per locomotion (cani robot, bipedi) si usa quasi sempre PPO con domain randomization in Isaac Lab. Per dexterous manipulation (es. il Dexterous Hand di OpenAI) SAC+HER è ancora il baseline di riferimento. Tutto passa per simulazione → transfer al robot reale.

SAC+HER su ambiente di simulazione robotica ha senso per la community scientifica non come contributo algoritmico finale, ma come **strumento di analisi**: è ancora il baseline standard per goal-conditioned manipulation, tutti i nuovi paper lo confrontano. Capire i suoi limiti — in particolare l'interazione con domain randomization — è direttamente utile. Il filone sim-to-real è inoltre più accessibile per ricerca accademica senza hardware costoso, e le domande aperte identificate in questo documento (HER+DR, ADR con transfer gap) si collocano esattamente all'intersezione tra i due mondi.

---

## Direzione 1 — HER + Domain Randomization: interazione durante il relabeling

### Domanda di ricerca

> "When HER relabels a transition, should the reward function use the original episode's physics parameters or freshly resampled ones?"

In un setup SAC+HER con domain randomization (es. massa dell'oggetto variabile), ogni episodio viene eseguito con parametri fisici campionati una volta. Quando HER rilabella le transizioni di quell'episodio, usa implicitamente il reward dell'ambiente — ma con quali parametri fisici? Quelli originali dell'episodio (comportamento di default in SB3) o ricampionati? Questa scelta non è mai stata studiata sistematicamente.

### Paper trovati

| Paper | Anno | Venue | Rilevanza |
|---|---|---|---|
| [Hindsight Experience Replay](https://arxiv.org/pdf/1707.01495) — Andrychowicz et al. | 2017 | NeurIPS | Paper originale HER. Nessuna DR. |
| [Solving Rubik's Cube with a Robot Hand](https://arxiv.org/abs/1910.07113) — Akkaya et al. (OpenAI) | 2019 | arXiv | Primo sistema a combinare HER + ADR. Non analizza la coerenza dei parametri fisici durante relabeling — scelta implicita mai giustificata. |
| [RoMo-HER: Robust Model-based HER](https://arxiv.org/html/2306.16061v1) | 2023 | arXiv | Usa modelli di dinamica appresi per generare goal virtuali. Affronta robustezza ma tramite modelli, non DR. |
| [Hindsight States: Blending Sim & Real Task Elements](https://arxiv.org/pdf/2303.02234) | 2023 | arXiv | Mescola elementi sim/real nel relabeling. Angolo diverso (sim-to-real blending), non studia coerenza fisica. |
| [GCHR: Goal-Conditioned Hindsight Regularization](https://arxiv.org/abs/2508.06108) | 2025 | arXiv | Argomenta che il relabeling HER standard non sfrutta appieno le esperienze. Non tocca DR o coerenza dei parametri fisici. |
| [MHER: Model-based Hindsight Experience Replay](https://arxiv.org/pdf/2107.00306) | 2021 | arXiv | Usa un modello di transizione per generare goal. Non studia DR. |

### Conclusione

**La domanda specifica è unstudied.** Il paper Akkaya et al. (2019) è l'unico lavoro rilevante che combina HER e DR in modo significativo, ma tratta i due moduli come separati e non analizza il design choice sulla coerenza dei parametri fisici durante il relabeling. Tutti i framework (SB3 incluso) usano i parametri originali dell'episodio per default, senza mai giustificarlo o confrontarlo con l'alternativa.

### Esperimenti da fare

1. **Baseline**: HER standard — relabeling con parametri fisici dell'episodio originale (default SB3)
2. **Variante A**: relabeling con parametri fisici ricampionati casualmente
3. **Variante B**: relabeling con i parametri del target domain fissi
4. Metriche: success rate su source, success rate su target, sample efficiency

---

## Direzione 2 — ADR con feedback dal transfer gap

### Domanda di ricerca

> "Can using the sim-to-real transfer gap (source SR − target SR) as the ADR curriculum signal improve transfer, compared to standard ADR which only uses training-domain success rate?"

L'ADR standard (Akkaya et al. 2019) espande/contrae i range di randomizzazione basandosi solo sul success rate nel dominio di training. Non sa nulla del target domain. L'idea è di usare invece la differenza di performance source-target come segnale diretto per guidare il curriculum.

### Paper trovati

| Paper | Anno | Venue | Rilevanza |
|---|---|---|---|
| [Solving Rubik's Cube with a Robot Hand](https://arxiv.org/abs/1910.07113) — Akkaya et al. | 2019 | arXiv | Definisce ADR standard. Baseline da battere. |
| [DORAEMON](https://arxiv.org/pdf/2311.01885) | 2024 | ICLR | Massimizza l'entropia della distribuzione di training mantenendo success rate ≥ soglia. Adatta la distribuzione sul solo dominio di training — nessun segnale dal target. Molto vicino ma non usa il gap. |
| [Auto-Tuned Sim-to-Real Transfer](https://arxiv.org/pdf/2104.07662) | 2021 | arXiv | Usa rollout sul robot reale per calibrare i parametri della sim (system identification). Usa dati reali ma per matching delle traiettorie, non come segnale di curriculum. |
| [Online vs Offline Adaptive DR Benchmark](https://arxiv.org/pdf/2206.14661) | 2022 | arXiv | Benchmark sistematico delle varianti ADR online e offline. Nessuna usa il transfer gap come segnale di espansione. |
| [Active Domain Randomization (SS-ADR)](https://arxiv.org/pdf/2002.07911) | 2020 | CoRL | Genera curriculum accoppiato goal-task. Usa differenze di performance tra ambienti come segnale, ma non il gap source-target in modo esplicito. |
| [Flow-based Domain Randomization (GoFlow)](https://arxiv.org/html/2502.01800v1) | 2025 | arXiv | Apprende distribuzioni di sampling con reti neurali (normalizing flows). Non usa il transfer gap. |
| [DROPO: Sim-to-Real with Offline DR](https://arxiv.org/pdf/2201.08434) | 2022 | Robotics & AS | Usa dimostrazioni offline reali per calibrare la distribuzione di randomizzazione. Usa dati reali ma per distribution matching, non curriculum. |
| [Understanding DR for Sim-to-Real](https://arxiv.org/abs/2110.03239) | 2021 | arXiv | Survey teorico. Fornisce bounds sul sim-to-real gap. Background utile. |

### Conclusione

**Lavoro adiacente esiste, ma il framing specifico è inesplorato.** DORAEMON è il più vicino — adatta la distribuzione di DR mantenendo un success rate minimo — ma usa solo il dominio di training. Nessun paper usa esplicitamente `(SR_source − SR_target)` come segnale di espansione/contrazione in un loop ADR. I metodi che usano dati reali (Auto-Tuned, DROPO) lo fanno per system identification, non come curriculum signal.

### Schema dell'algoritmo proposto

```
Loop ADR modificato:
  1. Esegui episodi boundary (come ADR standard)
  2. Valuta SR_source (success rate su source domain)
  3. Valuta SR_target (success rate su target domain, es. massa pesante fissa)
  4. Calcola gap = SR_source - SR_target
  5. Se gap > soglia_alta → il range attuale non trasferisce → contrai
     Se gap < soglia_bassa → il trasferimento è buono → espandi
     Altrimenti → mantieni
```

---

## Note generali sul panorama

- Il campo **HER + DR** è sorprendentemente poco studiato considerando che entrambe le tecniche sono standard per robotic manipulation
- La tendenza 2024-2025 è verso DR adattivo (DORAEMON, GoFlow) ma quasi sempre con segnale solo dal dominio di training
- I metodi che usano feedback dal target domain richiedono accesso al robot reale — un ADR che usa un **target domain simulato** come proxy è più pratico e non è stato proposto esplicitamente

---

## Target venue suggeriti

| Venue | Tipo | Scadenza indicativa |
|---|---|---|
| CoRL 2026 | Conference | ~giugno 2026 |
| NeurIPS 2026 Workshop | Workshop | ~settembre 2026 |
| ICRA 2027 | Conference | ~settembre 2026 |
| arXiv preprint | Preprint | anytime |

Per un primo paper, un **workshop CoRL o NeurIPS** è il target più realistico.
