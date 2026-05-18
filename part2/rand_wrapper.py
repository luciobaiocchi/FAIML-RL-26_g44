from collections import deque

import gymnasium as gym
import numpy as np


class RandomizationWrapper(gym.Wrapper):
    """
    Wrapper that randomizes the pushed object's mass at reset time.

    Modes:
    - `none`: keep the nominal environment mass unchanged
    - `udr`: sample uniformly from the full mass range
    - `adr`: sample from an adaptive range that expands/shrinks based on success
    """

    def __init__(
        self,
        env,
        mass_range=(1.0, 5.0),
        mode="none",
        adr_window=20,
        adr_high_threshold=0.8,
        adr_low_threshold=0.2,
        adr_step=0.25,
    ):
        super().__init__(env)

        if mode not in {"none", "udr", "adr"}:
            raise ValueError(f"Unsupported randomization mode: {mode}")

        mass_min_limit, mass_max_limit = mass_range
        if mass_min_limit <= 0 or mass_min_limit > mass_max_limit:
            raise ValueError(f"Invalid mass_range: {mass_range}")

        self.mode = mode
        self.mass_range = mass_range

        # Global limits allowed by domain randomization.
        self.mass_min_limit = float(mass_min_limit)
        self.mass_max_limit = float(mass_max_limit)

        # Nominal mass comes from the underlying environment definition.
        self.nominal_mass = float(self.env.unwrapped.task.current_mass)

        # Current ADR sampling range.
        self.mass_min = self.nominal_mass
        self.mass_max = self.nominal_mass

        # Keep a short success history to adapt the ADR range.
        self.success_history = deque(maxlen=adr_window)
        self.adr_window = adr_window
        self.adr_high_threshold = adr_high_threshold
        self.adr_low_threshold = adr_low_threshold
        self.adr_step = float(adr_step)

        self.last_sample_type = "fixed"
        self.last_mass = self.nominal_mass

        if self.mode == "udr":
            self.mass_min = self.mass_min_limit
            self.mass_max = self.mass_max_limit
        elif self.mode == "adr":
            # Start from the nominal mass and widen only when the policy succeeds.
            self.mass_min = self.nominal_mass
            self.mass_max = self.nominal_mass

    def _uniform(self, low: float, high: float) -> float:
        rng = getattr(self.env.unwrapped, "np_random", None)
        if rng is None:
            return float(np.random.uniform(low, high))
        return float(rng.uniform(low, high))

    def _sample_mass(self):
        if self.mode == "none":
            self.last_sample_type = "fixed"
            self.last_mass = self.nominal_mass
            return None

        if self.mode == "udr":
            self.last_sample_type = "uniform"
            self.last_mass = self._uniform(self.mass_min_limit, self.mass_max_limit)
            return self.last_mass

        if self.mode == "adr":
            self.last_sample_type = "adaptive"
            self.last_mass = self._uniform(self.mass_min, self.mass_max)
            return self.last_mass

        raise NotImplementedError(f"Sampling strategy '{self.mode}' is not implemented.")

    def _update_adr_range(self) -> None:
        if self.mode != "adr" or len(self.success_history) < self.adr_window:
            return

        success_rate = float(np.mean(self.success_history))
        self.success_history.clear()

        if success_rate >= self.adr_high_threshold:
            self.mass_min = max(self.mass_min_limit, self.mass_min - self.adr_step)
            self.mass_max = min(self.mass_max_limit, self.mass_max + self.adr_step)
        elif success_rate <= self.adr_low_threshold:
            self.mass_min = min(self.nominal_mass, self.mass_min + self.adr_step)
            self.mass_max = max(self.nominal_mass, self.mass_max - self.adr_step)

            if self.mass_min > self.mass_max:
                self.mass_min = self.nominal_mass
                self.mass_max = self.nominal_mass

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if self.mode == "adr" and (terminated or truncated):
            success = 0.0
            if isinstance(info, dict) and "is_success" in info:
                success = float(info["is_success"])
            self.success_history.append(success)
            self._update_adr_range()

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        new_mass = self._sample_mass()

        if new_mass is not None:
            sim = self.env.unwrapped.task.sim
            object_body_id = sim._bodies_idx["object"]

            sim.physics_client.changeDynamics(
                bodyUniqueId=object_body_id,
                linkIndex=-1,
                mass=float(new_mass),
            )

            print(
                f"[{self.mode}] mass={new_mass:.2f} "
                f"range=[{self.mass_min:.2f},{self.mass_max:.2f}] "
                f"type={self.last_sample_type}"
            )

        return super().reset(**kwargs)
