import math
import time

import gymnasium as gym
import numpy as np
import pybullet as p

from simple_driving.envs.simple_driving_env import SimpleDrivingEnv
from simple_driving.resources.car import Car
from simple_driving.resources.plane import Plane
from simple_driving.resources.goal import Goal
from simple_driving.resources.ramp import Ramp
from src.reward import RewardCalculator


class RallyDrivingEnv(SimpleDrivingEnv):
    """
    Rally environment: drive a fixed sequence of checkpoints as fast as possible.

    Scenarios:
        - phase1: bare track, just checkpoints
        - phase2: checkpoints + static cone obstacles to avoid
        - phase3: checkpoints + cones + ramps (jumps) for shortcuts
        - quiz2 scenarios (midpoint/none/random_pos): delegated to base SimpleDrivingEnv-style behaviour
    """

    # ── Observation bounds (kept from upstream) ────────────────────────
    OBS_RELX = 40.0
    OBS_RELY = 40.0
    OBS_OBSX = 40.0
    OBS_OBSY = 40.0
    OBS_PITCH = math.pi
    OBS_ROLL = math.pi
    OBS_VEL = 30.0
    OBS_YAW = 1.0

    # ── Default course ─────────────────────────────────────────────────
    CHECKPOINTS = [
        ( 16,  16),
        ( 16,   2),
        (  5,   4),
        ( -3,  12),
        (-16,  -1),
        (-16, -16),
    ]
    OBSTACLE_HOMES = [
        ( 8,  8),
        ( 0,  8),
        (-8,  0),
        ( 0, -8),
    ]
    RAMP_POSITIONS = [
        # (x, y, yaw_radians) — pointed roughly along the racing line
        (10, 9,  math.radians(-30)),
        (-8, 5,  math.radians( 120)),
    ]

    # Tunables
    CHECKPOINT_RADIUS = 1.5
    OBSTACLE_DRIFT_MAX = 0.05      # per substep position perturbation
    OBSTACLE_DRIFT_RADIUS = 2.0    # clamp drift from each obstacle's home
    OBSTACLE_COLLISION_RADIUS = 0.7

    def __init__(self, isDiscrete=False, renders=False,
                 reward_callback=None, observation_callback=None):
        super().__init__(
            isDiscrete=isDiscrete,
            renders=renders,
            reward_callback=reward_callback,
            observation_callback=observation_callback,
        )

        # 11-dimensional observation
        self.observation_space = gym.spaces.box.Box(
            low=np.array([
                -self.OBS_RELX, -self.OBS_RELY,        # relative goal x, y
                0.0,                                    # fraction of checkpoints remaining
                -self.OBS_OBSX, -self.OBS_OBSY,        # relative obstacle x, y (0 if none)
                0.0,                                    # has_obstacle flag
                -self.OBS_PITCH, -self.OBS_ROLL,
                -self.OBS_VEL,
                -self.OBS_YAW, -self.OBS_YAW,
            ], dtype=np.float32),
            high=np.array([
                self.OBS_RELX, self.OBS_RELY,
                1.0,
                self.OBS_OBSX, self.OBS_OBSY,
                1.0,
                self.OBS_PITCH, self.OBS_ROLL,
                self.OBS_VEL,
                self.OBS_YAW, self.OBS_YAW,
            ], dtype=np.float32),
        )

        # State
        self.prev_yaw = 0.0
        self.prev_roll = 0.0
        self.prev_pitch = 0.0
        self.prev_yaw_delta = 0.0
        self.checkpoints = []
        self.checkpoint_objects = []
        self.current_checkpoint_idx = 0

        self.obstacle_objects = []
        self.obstacle_positions = []
        self.obstacle_homes = []

        self.ramp_objects = []

        # Phase memory — useful for reward shaping (e.g. is jumping enabled this episode?)
        self.scenario = "phase1"

    # ── Reset ──────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super(SimpleDrivingEnv, self).reset(seed=seed)  # gym.Env.reset, not parent

        self._p.resetSimulation()
        self._p.setTimeStep(self._timeStep)
        self._p.setGravity(0, 0, -10)
        Plane(self._p)
        self.car = Car(self._p)
        self._envStepCounter = 0
        self.done = False
        self.reached_goal = False

        self.scenario = (options.get("scenario", "phase1") if options else "phase1")

        # Clear lists from previous episode
        self.obstacle_objects = []
        self.obstacle_positions = []
        self.obstacle_homes = []
        self.ramp_objects = []
        self.obstacle_pos = None
        self.has_obstacle = False

        if self.scenario == "phase1":
            pass  # bare track

        elif self.scenario == "phase2":
            self._spawn_obstacles()

        elif self.scenario == "phase3":
            self._spawn_obstacles()
            self._spawn_ramps()

        # else: empty track (also handles quiz scenarios falling through)

        # Checkpoints (allow override via options)
        checkpoints = options.get("checkpoints", None) if options else None
        self.checkpoints = checkpoints if checkpoints is not None else self.CHECKPOINTS
        self.checkpoint_objects = [Goal(self._p, pos) for pos in self.checkpoints]
        self.current_checkpoint_idx = 0

        car_pos = self.car.get_observation()
        self.prev_dist_to_goal = math.hypot(
            car_pos[0] - self.checkpoints[0][0],
            car_pos[1] - self.checkpoints[0][1],
        )
        _, car_orn = self._p.getBasePositionAndOrientation(self.car.car)
        euler = self._p.getEulerFromQuaternion(car_orn)
        self.prev_pitch, self.prev_roll, self.prev_yaw = euler

        ob = np.array(self.getExtendedObservation(), dtype=np.float32)
        return ob, {}

    def _spawn_obstacles(self):
        """Spawn the configured obstacle cones as visual-only markers
        (collision is distance-based in step())."""
        self.has_obstacle = True
        self.obstacle_homes = list(self.OBSTACLE_HOMES)
        self.obstacle_positions = list(self.OBSTACLE_HOMES)

        for home in self.obstacle_homes:
            vis_shape_id = self._p.createVisualShape(
                shapeType=p.GEOM_CYLINDER,
                radius=0.5, length=1.0,
                rgbaColor=[1.0, 0.5, 0.0, 1.0],
            )
            body_id = self._p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=vis_shape_id,
                basePosition=[home[0], home[1], 0.5],
            )
            self.obstacle_objects.append(body_id)

        # Used by parent class collision check — pick the closest one each step
        self.obstacle_pos = self.obstacle_homes[0]

    def _spawn_ramps(self):
        """Spawn ramps as physical bodies the car can drive over."""
        for (x, y, yaw) in self.RAMP_POSITIONS:
            self.ramp_objects.append(Ramp(self._p, (x, y), yaw=yaw))

    # ── Step ───────────────────────────────────────────────────────────

    def step(self, action):
        if self._isDiscrete:
            fwd = [-1, -1, -1, 0, 0, 0, 1, 1, 1]
            steerings = [-0.6, 0, 0.6, -0.6, 0, 0.6, -0.6, 0, 0.6]
            action = [fwd[action], steerings[action]]

        self.car.apply_action(action)

        # Track which obstacle (if any) is closest, for the reward callback
        nearest_obstacle = None

        for _ in range(self._actionRepeat):
            self._p.stepSimulation()
            if self._renders:
                time.sleep(self._timeStep)

            car_pos, car_orn = self._p.getBasePositionAndOrientation(self.car.car)
            pitch, roll, yaw = self._p.getEulerFromQuaternion(car_orn)

            # Dynamic obstacle wobble + collision check
            if self.obstacle_objects:
                self._wiggle_obstacles()
                nearest_obstacle, nearest_dist = self._nearest_obstacle(car_pos)
                if nearest_dist < self.OBSTACLE_COLLISION_RADIUS:
                    self.done = True
                    break

            if self._termination():
                self.done = True
                break
            self._envStepCounter += 1

        car_pos, car_orn = self._p.getBasePositionAndOrientation(self.car.car)
        pitch, roll, yaw = self._p.getEulerFromQuaternion(car_orn)
        current_goal = self.checkpoints[self.current_checkpoint_idx]

        dist_to_goal = math.hypot(
            car_pos[0] - current_goal[0],
            car_pos[1] - current_goal[1],
        )

        # out_of_bounds = abs(car_pos[0]) > 50 or abs(car_pos[1]) > 30
        # if self.done:
        #     print(f"Episode ended at step {self._envStepCounter}, pos={car_pos[:2]}, oob={out_of_bounds}")

        # Checkpoint logic
        self.reached_goal = False
        if dist_to_goal < self.CHECKPOINT_RADIUS:
            self.reached_goal = True
            self.current_checkpoint_idx += 1

            if self.current_checkpoint_idx >= len(self.checkpoints):
                self.done = True
            else:
                next_goal = self.checkpoints[self.current_checkpoint_idx]
                # Reset prev_dist so the progress signal isn't a huge jump
                dist_to_goal = math.hypot(
                    car_pos[0] - next_goal[0], car_pos[1] - next_goal[1],
                )
                self.prev_dist_to_goal = dist_to_goal
                current_goal = next_goal

        # Reward
        if self.reward_callback is None:
            raise ValueError("No reward_callback provided to RallyDrivingEnv.")

        reward = self.reward_callback(
            car_pos=car_pos,
            goal_pos=current_goal,
            obstacle_pos=nearest_obstacle,
            has_obstacle=self.has_obstacle,
            prev_dist_to_goal=self.prev_dist_to_goal,
            dist_to_goal=dist_to_goal,
            reached_goal=self.reached_goal,
            prev_yaw=self.prev_yaw, current_yaw=yaw,
            prev_yaw_delta=self.prev_yaw_delta,
            prev_roll=self.prev_roll, current_roll=roll,
            prev_pitch=self.prev_pitch, current_pitch=pitch,
            obstacle_positions=self.obstacle_positions,
            scenario=self.scenario,
        )

        current_yaw_delta = RewardCalculator._wrap_delta(yaw - self.prev_yaw)
        self.prev_yaw_delta = current_yaw_delta
        self.prev_yaw = yaw
        self.prev_roll = roll
        self.prev_pitch = pitch
        self.prev_dist_to_goal = dist_to_goal

        ob = np.array(self.getExtendedObservation(), dtype=np.float32)
        return ob, float(reward), self.done, False, {}

    # ── Helpers ────────────────────────────────────────────────────────

    def _wiggle_obstacles(self):
        """Apply small per-step drift to each obstacle, clamped within its home radius."""
        for idx, body_id in enumerate(self.obstacle_objects):
            home = self.obstacle_homes[idx]
            cx, cy = self.obstacle_positions[idx]

            dx = self.np_random.uniform(-self.OBSTACLE_DRIFT_MAX, self.OBSTACLE_DRIFT_MAX)
            dy = self.np_random.uniform(-self.OBSTACLE_DRIFT_MAX, self.OBSTACLE_DRIFT_MAX)
            new_x, new_y = cx + dx, cy + dy

            if math.hypot(new_x - home[0], new_y - home[1]) > self.OBSTACLE_DRIFT_RADIUS:
                new_x, new_y = cx - dx, cy - dy

            self.obstacle_positions[idx] = (new_x, new_y)
            self._p.resetBasePositionAndOrientation(
                body_id, [new_x, new_y, 0.5], [0, 0, 0, 1],
            )

    def _nearest_obstacle(self, car_pos):
        """Return (position, distance) of the nearest dynamic obstacle, or (None, inf)."""
        best = (None, float("inf"))
        for pos in self.obstacle_positions:
            d = math.hypot(car_pos[0] - pos[0], car_pos[1] - pos[1])
            if d < best[1]:
                best = (pos, d)
        return best

    # ── Observation ────────────────────────────────────────────────────

    def getExtendedObservation(self):
        car_pos, car_orn = self._p.getBasePositionAndOrientation(self.car.car)
        pitch, roll, yaw = self._p.getEulerFromQuaternion(car_orn)
        vel = self._p.getBaseVelocity(self.car.car)[0]
        speed = math.hypot(vel[0], vel[1])

        if self.current_checkpoint_idx < len(self.checkpoints):
            cur = self.checkpoints[self.current_checkpoint_idx]
        else:
            cur = self.checkpoints[-1]
        current_goal_3d = (cur[0], cur[1], 0.0)

        local_car_pos, local_car_orn = self._p.invertTransform(car_pos, car_orn)
        local_goal_pos, _ = self._p.multiplyTransforms(
            local_car_pos, local_car_orn, current_goal_3d, (0, 0, 0, 1),
        )

        if self.obstacle_positions:
            nearest, _ = self._nearest_obstacle(car_pos)
            obs_3d = (nearest[0], nearest[1], 0.0)
            local_obs_pos, _ = self._p.multiplyTransforms(
                local_car_pos, local_car_orn, obs_3d, (0, 0, 0, 1),
            )
        else:
            local_obs_pos = (0.0, 0.0, 0.0)

        checkpoints_remaining = (
            (len(self.checkpoints) - self.current_checkpoint_idx) / len(self.checkpoints)
        )

        return [
            local_goal_pos[0], local_goal_pos[1],
            checkpoints_remaining,
            local_obs_pos[0], local_obs_pos[1],
            float(self.has_obstacle),
            pitch, roll,
            speed,
            math.cos(yaw), math.sin(yaw),
        ]

    # ── Termination ────────────────────────────────────────────────────

    def _termination(self):
        car_pos, _ = self._p.getBasePositionAndOrientation(self.car.car)
        lap_complete = self.current_checkpoint_idx >= len(self.checkpoints)
        out_of_bounds = abs(car_pos[0]) > 50 or abs(car_pos[1]) > 30
        return self._envStepCounter > 500 or lap_complete or out_of_bounds
