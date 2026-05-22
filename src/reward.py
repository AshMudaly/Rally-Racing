import numpy as np


class RewardCalculator:
    """
    Reward function for the oval track racing agent.

    Components:
        + progress_reward   — SIGNED forward movement (in body frame)
        + speed_reward      — incentivise going fast
        - time_penalty      — small per-step cost so standing still bleeds reward
        - swerve_penalty    — penalise rapid steering changes
        - wall_proximity    — penalise getting too close to walls
        - off_track_penalty — large negative if car leaves track
    """

    def __init__(self):
        # Weights — tuned after first run showed agent learned "do nothing"
        self.w_progress      = 5.0    # was 2.0 — make forward motion dominant
        self.w_speed         = 0.5
        self.w_swerve        = 0.3    # was 0.8 — too harsh, agent refused to steer
        self.w_wall          = 0.3
        self.w_collision     = 50.0
        self.w_time          = 0.05   # NEW — constant per-step penalty

        # Thresholds
        self.min_ray_distance  = 0.3
        self.crash_distance    = 0.05
        self.max_ray_distance  = 5.0

        # Going backwards is worse than standing still
        self.reverse_multiplier = 2.0

        self.prev_steering = 0.0
        self.prev_position = None
        self.total_reward  = 0.0

    def reset(self):
        self.prev_steering = 0.0
        self.prev_position = None
        self.total_reward  = 0.0

    def compute(
        self,
        rays: np.ndarray,
        speed: float,
        steering: float,
        position: np.ndarray,
        yaw: float,
        crashed: bool,
    ) -> tuple[float, dict]:
        reward = 0.0
        info   = {}

        # ── 1. Signed forward progress ──────────────────────────────────
        if self.prev_position is not None:
            delta   = position - self.prev_position
            heading = np.array([np.cos(yaw), np.sin(yaw)], dtype=np.float32)
            forward = float(np.dot(delta, heading))

            if forward >= 0.0:
                progress_r = self.w_progress * forward
            else:
                progress_r = self.w_progress * forward * self.reverse_multiplier
        else:
            progress_r = 0.0
        reward += progress_r
        info["progress"] = progress_r

        # ── 2. Speed reward ─────────────────────────────────────────────
        speed_r = self.w_speed * min(speed, 3.0) / 3.0
        reward += speed_r
        info["speed"] = speed_r

        # ── 3. Swerve penalty ───────────────────────────────────────────
        steering_delta = abs(steering - self.prev_steering)
        swerve_p = -self.w_swerve * (steering_delta ** 2)
        reward += swerve_p
        info["swerve_penalty"] = swerve_p

        # ── 4. Wall proximity penalty ───────────────────────────────────
        min_ray = float(np.min(rays))
        if min_ray < self.min_ray_distance:
            wall_p = -self.w_wall * (
                (self.min_ray_distance - min_ray) / self.min_ray_distance
            ) ** 2
        else:
            wall_p = 0.0
        reward += wall_p
        info["wall_proximity"] = wall_p

        # ── 5. Collision penalty ────────────────────────────────────────
        if crashed:
            collision_p = -self.w_collision
            reward += collision_p
            info["collision"] = collision_p
        else:
            info["collision"] = 0.0

        # ── 6. Time penalty ─────────────────────────────────────────────
        # Constant per-step cost so the "stand still for 1000 steps" strategy
        # accrues -50 reward, same magnitude as a crash. Forces the agent to
        # earn reward by moving rather than passively avoiding penalties.
        time_p = -self.w_time
        reward += time_p
        info["time_penalty"] = time_p

        # ── Bookkeeping ─────────────────────────────────────────────────
        self.prev_steering  = steering
        self.prev_position  = position.copy()
        self.total_reward  += reward

        info["total_step"]    = reward
        info["episode_total"] = self.total_reward

        return reward, info

    def is_crashed(self, rays: np.ndarray) -> bool:
        return bool(np.any(rays * self.max_ray_distance < self.crash_distance))

    def normalise_rays(self, raw_rays: np.ndarray) -> np.ndarray:
        return np.clip(raw_rays / self.max_ray_distance, 0.0, 1.0)