"""
Reward function for the RallyDrivingEnv.

Goals:
    + Reward forward progress toward the next checkpoint
    + Reward reaching checkpoints
    - Penalise standing still (step penalty)
    - Penalise hitting obstacles
    - Penalise out-of-bounds
    - Penalise sudden orientation changes (roll, pitch, yaw)
    - Penalise driving close to obstacles (repulsive field)
    +/- Reward or penalise landing big jumps depending on weight signs

The function is exposed both as a class (cleaner for tests / per-instance state)
and as a module-level `custom_reward` callable for backwards compatibility with
`SimpleDrivingEnv`'s callback interface.
"""

import math


class RewardConfig:
    """All tunable weights and thresholds in one place."""

    # ── Major events ───────────────────────────────────────────────────
    GOAL_REWARD       = 100.0    # reaching a checkpoint
    OBSTACLE_PENALTY  = -100.0   # close-contact with an obstacle
    OUT_OF_BOUNDS     = -50.0
    WORLD_BOUNDARY    =  30.0    # half-extent of the play area

    # ── Per-step shaping ───────────────────────────────────────────────
    STEP_PENALTY        =  -2.0
    PROGRESS_SCALE      =   5.0    # multiplier on (prev_dist - dist)
    # FIX: regression penalty now fires unconditionally on backward movement,
    # not only when near an obstacle. Previously the guard meant phase1 had
    # zero regression penalty, letting the agent zigzag freely.
    REGRESSION_PENALTY  = -10.0

    # ── Smooth driving ─────────────────────────────────────────────────
    # FIX: raised from -1.0 to -8.0. At -1.0 even tiny forward progress
    # (~0.2 units/step * PROGRESS_SCALE=5 = +1.0) outweighed the jerk
    # penalty, making oscillation profitable. At -8.0 the agent must
    # maintain a straight line to earn positive reward.
    YAW_JERK_PENALTY    =  -8.0   # per radian of yaw rate change
    ROLL_DELTA_PENALTY  = -15.0   # per radian of roll change (chassis tilt)
    PITCH_DELTA_PENALTY =  -4.0   # per radian of pitch change

    # ── Obstacle handling ──────────────────────────────────────────────
    MIN_SAFE_DISTANCE     = 1.0    # closer than this is a "hit"
    REPULSE_RADIUS        = 2.5    # range of repulsive field
    REPULSE_SCALE         = 10.0   # strength of repulsive field

    # ── Jump bonus (phase 3) ───────────────────────────────────────────
    # Positive value => agent is rewarded for getting airborne briefly.
    # Set to 0.0 or negative to discourage jumping. Triggered by pitch
    # magnitude exceeding the threshold (i.e. the car launched off a ramp).
    AIRBORNE_PITCH_THRESHOLD = 0.20   # radians — about 11 degrees
    AIRBORNE_BONUS           = 1.0    # per step while pitched up


class RewardCalculator:
    """
    Computes per-step reward. Stateless between calls — the env provides
    all prev_* fields. Just wraps `RewardConfig` to keep the code testable.
    """

    def __init__(self, config: RewardConfig | None = None):
        self.cfg = config or RewardConfig()

    def __call__(
        self,
        car_pos, goal_pos, obstacle_pos, has_obstacle,
        prev_dist_to_goal, dist_to_goal, reached_goal,
        prev_yaw=0.0, current_yaw=0.0,
        prev_yaw_delta=0.0,
        prev_roll=0.0, current_roll=0.0,
        prev_pitch=0.0, current_pitch=0.0,
        obstacle_positions=None, scenario="phase1",
    ) -> float:
        cfg = self.cfg
        reward = 0.0

        # ── Progress ────────────────────────────────────────────────────
        progress = (prev_dist_to_goal - dist_to_goal) if prev_dist_to_goal is not None else 0.0
        reward += cfg.STEP_PENALTY + cfg.PROGRESS_SCALE * progress

        # ── Regression: penalise moving away from goal (all scenarios) ──
        # FIX: removed has_obstacle guard — regression must fire in phase1
        # too, otherwise the agent has no cost for zigzagging.
        if progress < 0:
            reward += cfg.REGRESSION_PENALTY

        # ── Checkpoint reached ──────────────────────────────────────────
        if reached_goal:
            reward += cfg.GOAL_REWARD

        # ── Yaw smoothness ──────────────────────────────────────────────
        yaw_delta = self._wrap_delta(current_yaw - prev_yaw)
        yaw_jerk  = self._wrap_delta(yaw_delta - prev_yaw_delta)
        reward += cfg.YAW_JERK_PENALTY * abs(yaw_jerk)

        # ── Chassis stability (roll/pitch) ──────────────────────────────
        roll_delta  = self._wrap_delta(current_roll - prev_roll)
        pitch_delta = self._wrap_delta(current_pitch - prev_pitch)
        reward += cfg.ROLL_DELTA_PENALTY * abs(roll_delta)
        reward += cfg.PITCH_DELTA_PENALTY * abs(pitch_delta)

        # ── Obstacle hit (single nearest) ───────────────────────────────
        if has_obstacle and obstacle_pos is not None:
            dist_to_obs = math.hypot(
                car_pos[0] - obstacle_pos[0], car_pos[1] - obstacle_pos[1],
            )
            if dist_to_obs <= cfg.MIN_SAFE_DISTANCE:
                reward += cfg.OBSTACLE_PENALTY

        # ── Obstacle repulsive field (all obstacles) ────────────────────
        if obstacle_positions:
            for pos in obstacle_positions:
                d = math.hypot(car_pos[0] - pos[0], car_pos[1] - pos[1])
                if d < cfg.REPULSE_RADIUS:
                    reward -= cfg.REPULSE_SCALE * (cfg.REPULSE_RADIUS - d)

        # ── Out of bounds ───────────────────────────────────────────────
        if abs(car_pos[0]) > cfg.WORLD_BOUNDARY or abs(car_pos[1]) > cfg.WORLD_BOUNDARY:
            reward += cfg.OUT_OF_BOUNDS

        # ── Airborne bonus (phase 3 jumps) ──────────────────────────────
        if scenario == "phase3" and current_pitch > cfg.AIRBORNE_PITCH_THRESHOLD and progress > 0:
            reward += cfg.AIRBORNE_BONUS

        return reward

    @staticmethod
    def _wrap_delta(delta: float) -> float:
        """Wrap an angular difference into [-pi, pi]."""
        if delta > math.pi:
            return delta - 2 * math.pi
        if delta < -math.pi:
            return delta + 2 * math.pi
        return delta


# ── Module-level callable for env.reward_callback ──────────────────────
_default_calculator = RewardCalculator()


def custom_reward(**kwargs) -> float:
    return _default_calculator(**kwargs)