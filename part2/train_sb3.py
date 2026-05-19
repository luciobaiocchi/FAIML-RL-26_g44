import argparse
import sys

import gymnasium as gym
import panda_gym  # type: ignore[import-not-found]
from rand_wrapper import RandomizationWrapper
from wrappers import RewardShapingWrapper
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer


# ─── Parametri specifici per algoritmo ──────────────────────────────────────

PPO_ONLY = {"n_envs", "ent_coef", "n_steps", "clip_range", "gae_lambda"}
SAC_ONLY = {"her", "batch_size", "gradient_steps",
            "learning_starts", "buffer_size", "tau", "train_freq", "device"}
# learning_rate è condiviso tra PPO e SAC


# ─── Parsing e validazione ────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PPO or SAC on PandaPush-v3",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Argomenti comuni
    parser.add_argument("--algorithm", type=str, required=True, choices=["ppo", "sac"])
    parser.add_argument("--sampling-strategy", type=str, default="none",
                        choices=["none", "udr", "adr"])
    parser.add_argument("--env-type", type=str, default="source",
                        choices=["source", "target"])
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", action="store_true", default=False)
    parser.add_argument("--run-name", type=str, default=None,
                        help="Nome file del modello salvato (override del nome automatico)")
    parser.add_argument("--learning-rate", type=float, default=None,
                        help="[PPO/SAC] Learning rate")

    # PPO only
    parser.add_argument("--n-envs", type=int, default=None,
                        help="[PPO] Numero di env paralleli (>1 usa SubprocVecEnv)")
    parser.add_argument("--ent-coef", type=float, default=None,
                        help="[PPO] Coefficiente entropico — forza l'esplorazione")
    parser.add_argument("--n-steps", type=int, default=None,
                        help="[PPO] Step raccolti per env prima di ogni update")
    parser.add_argument("--clip-range", type=float, default=None,
                        help="[PPO] Clipping del ratio policy (epsilon in PPO-clip)")
    parser.add_argument("--gae-lambda", type=float, default=None,
                        help="[PPO] Lambda per Generalized Advantage Estimation")

    # SAC only
    parser.add_argument("--her", action="store_true", default=None,
                        help="[SAC] Abilita Hindsight Experience Replay")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="[SAC] Dimensione del minibatch")
    parser.add_argument("--gradient-steps", type=int, default=None,
                        help="[SAC] Update per step raccolto (-1 = stesso numero di step)")
    parser.add_argument("--learning-starts", type=int, default=None,
                        help="[SAC] Step prima di iniziare il training")
    parser.add_argument("--buffer-size", type=int, default=None,
                        help="[SAC] Dimensione del replay buffer")
    parser.add_argument("--tau", type=float, default=None,
                        help="[SAC] Soft update coefficient per i target network")
    parser.add_argument("--train-freq", type=int, default=None,
                        help="[SAC] Step da raccogliere prima di ogni ciclo di training")
    parser.add_argument("--device", type=str, default=None,
                        choices=["auto", "cpu", "cuda", "mps"],
                        help="[SAC] Device (mps per Apple Silicon, cuda per NVIDIA)")

    args = parser.parse_args()
    _validate_and_set_defaults(args)
    return args


def _validate_and_set_defaults(args: argparse.Namespace) -> None:
    if args.algorithm == "ppo":
        # Warn se passati flag SAC-only
        sac_passed = []
        if args.her:
            sac_passed.append("--her")
        for param in ["batch_size", "gradient_steps", "learning_starts",
                      "buffer_size", "tau", "train_freq", "device"]:
            if getattr(args, param) is not None:
                sac_passed.append(f"--{param.replace('_', '-')}")
        if sac_passed:
            print(f"[WARNING] Parametri SAC-only ignorati con --algorithm ppo: "
                  f"{', '.join(sac_passed)}", file=sys.stderr)
        # Default PPO
        if args.learning_rate is None:
            args.learning_rate = 3e-4
        if args.n_envs is None:
            args.n_envs = 1
        if args.ent_coef is None:
            args.ent_coef = 0.001
        if args.n_steps is None:
            args.n_steps = 2048
        if args.clip_range is None:
            args.clip_range = 0.2
        if args.gae_lambda is None:
            args.gae_lambda = 0.95

    elif args.algorithm == "sac":
        # Warn se passati flag PPO-only
        ppo_passed = []
        if args.n_envs is not None:
            ppo_passed.append("--n-envs")
        for param in ["ent_coef", "n_steps", "clip_range", "gae_lambda"]:
            if getattr(args, param) is not None:
                ppo_passed.append(f"--{param.replace('_', '-')}")
        if ppo_passed:
            print(f"[WARNING] Parametri PPO-only ignorati con --algorithm sac: "
                  f"{', '.join(ppo_passed)}", file=sys.stderr)
        # Default SAC
        if args.learning_rate is None:
            args.learning_rate = 3e-4
        if args.her is None:
            args.her = False
        if args.batch_size is None:
            args.batch_size = 256
        if args.gradient_steps is None:
            args.gradient_steps = 1
        if args.learning_starts is None:
            args.learning_starts = 100
        if args.buffer_size is None:
            args.buffer_size = 1_000_000
        if args.tau is None:
            args.tau = 0.005
        if args.train_freq is None:
            args.train_freq = 1
        if args.device is None:
            args.device = "auto"


# ─── Costruzione ambiente ─────────────────────────────────────────────────────

def make_env(env_type: str, sampling_strategy: str, seed: int, rank: int = 0):
    def _init():
        env = gym.make(
            "PandaPush-v3",
            render_mode="rgb_array",
            max_episode_steps=500,
            type=env_type,
            reward_type="dense",
        )
        env = RandomizationWrapper(env, mass_range=(1.0, 5.0), mode=sampling_strategy)
        env = RewardShapingWrapper(env, bonus_distance=0.05, bonus=1.0, time_penalty=1e-2)
        env.reset(seed=seed + rank)
        return env
    return _init


# ─── Training PPO ─────────────────────────────────────────────────────────────

def train_ppo(args: argparse.Namespace) -> None:
    env_fns = [make_env(args.env_type, args.sampling_strategy, args.seed, i)
               for i in range(args.n_envs)]
    env = SubprocVecEnv(env_fns) if args.n_envs > 1 else DummyVecEnv(env_fns)
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)

    model = PPO(
        "MultiInputPolicy",
        env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        gamma=0.99,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        gae_lambda=args.gae_lambda,
        verbose=1,
        tensorboard_log="./ppo_logs/",
        seed=args.seed,
        policy_kwargs=dict(net_arch=[256, 256]),
    )
    model.learn(total_timesteps=args.timesteps, log_interval=1)

    if args.save:
        from pathlib import Path
        Path("models").mkdir(exist_ok=True)
        base = args.run_name or f"ppo_{args.sampling_strategy}_{args.env_type}_{args.timesteps // 1000}k"
        name = f"models/{base}"
        model.save(name)
        env.save(name + "_vecnorm.pkl")
        print(f"Modello salvato: {name}.zip")


# ─── Training SAC ─────────────────────────────────────────────────────────────

def train_sac(args: argparse.Namespace) -> None:
    env = make_env(args.env_type, args.sampling_strategy, args.seed)()

    replay_buffer_class = HerReplayBuffer if args.her else None
    replay_buffer_kwargs = dict(
        n_sampled_goal=4,
        goal_selection_strategy="future",
    ) if args.her else None

    model = SAC(
        "MultiInputPolicy",
        env,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_steps=args.gradient_steps,
        learning_starts=args.learning_starts,
        buffer_size=args.buffer_size,
        tau=args.tau,
        train_freq=args.train_freq,
        ent_coef="auto_0.1",
        verbose=1,
        tensorboard_log="./sac_logs/",
        seed=args.seed,
        device=args.device,
        replay_buffer_class=replay_buffer_class,
        replay_buffer_kwargs=replay_buffer_kwargs,
    )
    model.learn(total_timesteps=args.timesteps, log_interval=4)

    if args.save:
        from pathlib import Path
        Path("models").mkdir(exist_ok=True)
        base = args.run_name or (f"sac_{args.sampling_strategy}_{args.env_type}"
                                 f"_{args.timesteps // 1000}k{'_her' if args.her else ''}")
        name = f"models/{base}"
        model.save(name)
        print(f"Modello salvato: {name}.zip")


# ─── Entry point ─────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    print(f"Algorithm : {args.algorithm.upper()}")
    print(f"Strategy  : {args.sampling_strategy} | Env: {args.env_type} | "
          f"Steps: {args.timesteps} | Seed: {args.seed}")
    if args.algorithm == "ppo":
        print(f"PPO       : n_envs={args.n_envs} | lr={args.learning_rate} | "
              f"ent_coef={args.ent_coef} | n_steps={args.n_steps} | "
              f"clip={args.clip_range} | gae_lambda={args.gae_lambda}")
    else:
        print(f"SAC       : her={args.her} | lr={args.learning_rate} | "
              f"batch={args.batch_size} | grad_steps={args.gradient_steps} | "
              f"device={args.device}")

    if args.algorithm == "ppo":
        train_ppo(args)
    else:
        train_sac(args)


if __name__ == "__main__":
    main()