import argparse
from collections import deque

import gymnasium as gym
import numpy as np
import panda_gym  # type: ignore[import-not-found]
from rand_wrapper import RandomizationWrapper

from stable_baselines3 import PPO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on PandaPush-v3")
    parser.add_argument(
        "--sampling-strategy",
        type=str,
        default="none",
        choices=["none", "udr", "adr"],
        help="Sampling strategy for the object mass",
    )
    parser.add_argument(
        "--env-type",
        type=str,
        default="source",
        choices=["source", "target"],
        help="PandaPush environment type",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=500_000,
        help="Number of training timesteps",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for environment and model",
    )
    parser.add_argument(
        "--save",
        type=bool,
        default=False,
        choices=[True, False],
        help="Select if you want to save or not the model weights"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    env = gym.make(
        "PandaPush-v3",
        render_mode="rgb_array",
        type=args.env_type,
        reward_type="dense",
    )

    env = RandomizationWrapper(
        env,
        mass_range=(1.0, 5.0),
        mode=args.sampling_strategy,
    )
    env.reset(seed=args.seed)

    model = PPO(
        "MultiInputPolicy",
        env,
        learning_rate=1e-4,
        n_steps=1024,
        gamma=0.99,
        clip_range=0.2,
        ent_coef=0.005,
        verbose=1,
        tensorboard_log="./ppo_logs/",
        seed=args.seed,
    )


    model.learn(total_timesteps=args.timesteps, log_interval=1)
    if args.save:
        save_name = f"ppo_push_{args.sampling_strategy}_{args.env_type}_{args.timesteps // 1000}k"
        model.save(save_name)


if __name__ == "__main__":
    main()
