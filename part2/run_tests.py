import argparse
import itertools
import subprocess
import sys
from datetime import datetime
from pathlib import Path


# ─── Parameter grids per profilo ────────────────────────────────────────────
#
# Ogni profilo definisce un sottoinsieme di parametri da esplorare.
# Lo script genera tutte le combinazioni (prodotto cartesiano) e le esegue.

SAC_PROFILES: dict[int, dict] = {
    1: {  # Baseline: learning rate vs batch size, no HER, no DR
        "learning_rate": [3e-4, 1e-3],
        "batch_size":    [256, 512],
        "her":           [False],
        "sampling_strategy": ["none"],
    },
    2: {  # Gradient steps: quanti update per step raccolto
        "learning_rate":   [3e-4, 1e-3],
        "gradient_steps":  [1, 4, 8],
        "her":             [False],
        "sampling_strategy": ["none"],
    },
    3: {  # Impatto di HER con e senza domain randomization
        "learning_rate":   [3e-4, 1e-3],
        "her":             [False],
        "sampling_strategy": ["none", "udr"],
    },
    4: {  # Combo: batch size + gradient steps + HER + DR
        "learning_rate":   [3e-4],
        "batch_size":      [256, 512],
        "gradient_steps":  [1, 4],
        "her":             [False],
        "sampling_strategy": ["none", "udr"],
    },
}

PPO_PROFILES: dict[int, dict] = {
    1: {  # Learning rate vs ent_coef — i due iperparametri più impattanti
        "learning_rate":   [3e-4, 1e-3],
        "ent_coef":        [0.001, 0.005],
        "n_envs":          [4],
        "sampling_strategy": ["none"],
    },
    2: {  # n_steps vs clip_range — dimensione rollout e aggressività update
        "n_steps":         [1024, 2048],
        "clip_range":      [0.1, 0.2],
        "n_envs":          [4],
        "sampling_strategy": ["none"],
    },
    3: {  # Iperparametri + domain randomization
        "learning_rate":   [3e-4, 1e-3],
        "ent_coef":        [0.001, 0.005],
        "n_envs":          [4],
        "sampling_strategy": ["none", "udr"],
    },
    4: {  # Full combo: lr + n_steps + DR
        "learning_rate":   [3e-4, 1e-3],
        "n_steps":         [1024, 2048],
        "ent_coef":        [0.001],
        "n_envs":          [4],
        "sampling_strategy": ["none", "udr"],
    },
}

# Mappatura nome parametro → flag CLI
SAC_FLAG_MAP: dict[str, str] = {
    "learning_rate":    "--learning-rate",
    "batch_size":       "--batch-size",
    "gradient_steps":   "--gradient-steps",
    "her":              "--her",          # store_true
    "sampling_strategy": "--sampling-strategy",
}

PPO_FLAG_MAP: dict[str, str] = {
    "learning_rate":    "--learning-rate",
    "ent_coef":         "--ent-coef",
    "n_steps":          "--n-steps",
    "clip_range":       "--clip-range",
    "gae_lambda":       "--gae-lambda",
    "n_envs":           "--n-envs",
    "sampling_strategy": "--sampling-strategy",
}

# Timesteps per smoke test (piccoli ma validi per entrambi gli algoritmi)
SMOKE_TIMESTEPS = {
    "sac": 500,    # supera learning_starts=100, fa qualche update
    "ppo": 3000,   # circa 1 rollout con n_steps=2048; con n_envs>1 ne fa comunque 1 completo
}

DEFAULT_TIMESTEPS = 500_000


# ─── Utility ────────────────────────────────────────────────────────────────

def grid_combinations(grid: dict) -> list[dict]:
    keys = list(grid.keys())
    return [
        dict(zip(keys, combo))
        for combo in itertools.product(*grid.values())
    ]


def make_run_name(algorithm: str, params: dict) -> str:
    parts = [algorithm]
    for k, v in params.items():
        if isinstance(v, float):
            parts.append(f"{k}{v:.0e}")
        elif isinstance(v, bool):
            parts.append(f"{k}{int(v)}")
        else:
            parts.append(f"{k}{v}")
    return "_".join(parts)


def build_command(algorithm: str, params: dict, timesteps: int, save: bool, run_name: str) -> list[str]:
    flag_map = SAC_FLAG_MAP if algorithm == "sac" else PPO_FLAG_MAP
    cmd = [sys.executable, "train_sb3.py",
        "--algorithm", algorithm,
        "--timesteps", str(timesteps),
        "--env-type", "source",
        *( ["--save"] if save else [] ),
        "--run-name", run_name,
    ]
    for key, value in params.items():
        flag = flag_map.get(key)
        if flag is None:
            continue
        if isinstance(value, bool):
            if value:
                cmd.append(flag)   # store_true: aggiunge il flag solo se True
        else:
            cmd.extend([flag, str(value)])
    return cmd


# ─── Runner ─────────────────────────────────────────────────────────────────

def run_experiment(cmd: list[str], output_dir: Path, run_name: str) -> bool:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "training_log.txt"

    print(f"\n{'─'*60}")
    print(f"  {run_name}")
    print(f"  {' '.join(cmd)}")
    print(f"{'─'*60}")

    with open(log_file, "w") as f:
        f.write(f"run:       {run_name}\n")
        f.write(f"command:   {' '.join(cmd)}\n")
        f.write(f"timestamp: {datetime.now().isoformat()}\n")
        f.write("─" * 60 + "\n\n")

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        f.write(result.stdout)

    ok = result.returncode == 0
    status = "OK" if ok else f"FAILED (exit {result.returncode})"
    print(f"  → {status}  (log: {log_file})")

    # Mostra le ultime righe dell'output per visibilità immediata
    tail = result.stdout.strip().splitlines()[-6:]
    for line in tail:
        print(f"     {line}")

    return ok


# ─── Main ───────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grid search su parametri SAC/PPO, diviso in 4 profili."
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        required=True,
        choices=["sac", "ppo"],
        help="Algoritmo da usare",
    )
    parser.add_argument(
        "--profile",
        type=int,
        required=True,
        choices=[1, 2, 3, 4],
        help="Profilo parametri (1-4)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        default=False,
        help="Smoke test: esegue ogni run con pochi timesteps per verificare che non crashino",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        default=False,
        help="Salva il modello al termine di ogni run",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    profiles = SAC_PROFILES if args.algorithm == "sac" else PPO_PROFILES
    grid = profiles[args.profile]
    combinations = grid_combinations(grid)
    timesteps = SMOKE_TIMESTEPS[args.algorithm] if args.smoke else DEFAULT_TIMESTEPS
    base_dir = Path("test") / f"profile_{args.profile}"

    print(f"\nAlgoritmo : {args.algorithm.upper()}")
    print(f"Profilo   : {args.profile}")
    print(f"Smoke     : {args.smoke}")
    print(f"Timesteps : {timesteps}")
    print(f"Run totali: {len(combinations)}")
    print(f"Output    : {base_dir}/")

    results: list[tuple[str, bool]] = []

    for i, params in enumerate(combinations, 1):
        run_name = make_run_name(args.algorithm, params)
        output_dir = base_dir / run_name
        cmd = build_command(args.algorithm, params, timesteps, args.save, run_name)
        print(f"\n[{i}/{len(combinations)}]")
        ok = run_experiment(cmd, output_dir, run_name)
        results.append((run_name, ok))

    # Riepilogo finale
    print(f"\n{'='*60}")
    print(f"RIEPILOGO — {len(results)} run")
    print(f"{'='*60}")
    passed = sum(ok for _, ok in results)
    for run_name, ok in results:
        icon = "✓" if ok else "✗"
        print(f"  {icon}  {run_name}")
    print(f"\n{passed}/{len(results)} completate con successo.")
    print(f"Log salvati in: {base_dir}/")


if __name__ == "__main__":
    main()
