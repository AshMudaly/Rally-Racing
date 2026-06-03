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


def _wrap_delta(delta: float) -> float:
    """Wrap an angular difference into [-pi, pi]."""
    if delta > math.pi:
        return delta - 2 * math.pi
    if delta < -math.pi:
        return delta + 2 * math.pi
    return delta


class RallyDrivingEnv(SimpleDrivingEnv):
    """
    Rally environment: drive a fixed sequence of checkpoints as fast as possible.

    Circuit curriculum (counter-clockwise oval, ~24m x 16m):
        - circuit_easy:      checkpoints only
        - circuit_medium:    checkpoints + ramps
        - circuit_hard:      checkpoints + ramps + 4 SW-cluster cones
        - circuit_difficult: checkpoints + ramps + all 8 cones

    Obstacles spawn with per-reset jitter (uniform disc of radius
    OBSTACLE_SPAWN_JITTER around their nominal home) so the policy cannot
    memorise a fixed layout. Within an episode obstacles are static.
    """

    # ── Observation bounds ─────────────────────────────────────────────
    OBS_RELX  = 40.0
    OBS_RELY  = 40.0
    OBS_OBSX  = 40.0
    OBS_OBSY  = 40.0
    OBS_PITCH = math.pi
    OBS_ROLL  = math.pi
    OBS_VEL   = 30.0
    OBS_YAW   = 1.0

    # ── Camera (car-mounted, yaw-following) ────────────────────────────
    CAM_WIDTH      = 84
    CAM_HEIGHT     = 84
    CAM_FOV        = 55.0
    CAM_EYE_HEIGHT = 0.50   # camera height above car origin (m)
    CAM_EYE_FWD    = 0.30   # forward offset from car origin (m)
    CAM_TARGET_FWD = 4.0    # look-at point distance ahead (m)
    CAM_NEAR       = 0.05
    CAM_FAR        = 50.0

    # ── Circuit course (counter-clockwise oval) ────────────────────────
    CIRCUIT_CHECKPOINTS = [
        (  0, -14),   # CP1  start/finish, bottom center
        (-10, -14),   # CP2  after second ramp (gap pair)
        (-16,  -2),   # CP3  cone slalom / big-radius left
        (  0,  14),   # CP4  long straight, top
        ( 14,   8),   # CP5  gentle right
        ( 16,  -4),   # CP6  sweeping left
        ( 15, -14),   # CP7  before single ramp
        ( -6, -14),   # CP8  past ramps, closes the lap
    ]
    # Order matters: first 4 are the SW cluster used by circuit_hard;
    # entries 4..7 are the additional cones added in circuit_difficult.
    CIRCUIT_OBSTACLE_HOMES = [
        (-15, -9),
        (-10, -11),
        (-12, -6),
        (-14, -5),
        (  9,  8),
        ( 11, 11),
        ( -7,  5),
        ( -4, 10),
    ]
    CIRCUIT_RAMP_POSITIONS = [
        (10, -14, math.radians(180)),
        (-2, -14, math.radians(180)),
        (-4, -14, math.radians(0)),
        (15,  2, math.radians(270)),
    ]
    CIRCUIT_HARD_OBSTACLE_COUNT = 4

    # ── Tunables ───────────────────────────────────────────────────────
    CHECKPOINT_RADIUS         = 1.5
    OBSTACLE_SPAWN_JITTER     = 1.5    # per-reset spawn radius around home
    OBSTACLE_COLLISION_RADIUS = 0.7

    # ── Ramp-taken detection ──────────────────────────────────────────
    RAMP_TAKEN_RADIUS  = 1.5     # car must be within this many metres of ramp centre
    RAMP_PITCH_THRESH  = 0.15    # car pitch must exceed this (≈9°) to count as airborne
    FLIP_THRESHOLD     = 2.5     # radians; ~143° — beyond any legitimate ramp pitch

    # ── Scenario membership sets (single source of truth) ──────────────
    _SCENARIOS_WITH_OBSTACLES = ("circuit_hard", "circuit_difficult")
    _SCENARIOS_WITH_RAMPS     = ("circuit_medium", "circuit_hard",
                                 "circuit_difficult")

    def __init__(self, isDiscrete=False, renders=False,
                 reward_callback=None, observation_callback=None,
                 scenario="circuit_easy"):
        super().__init__(
            isDiscrete=isDiscrete,
            renders=renders,
            reward_callback=reward_callback,
            observation_callback=observation_callback,
        )

        # 11-dimensional observation space
        self.observation_space = gym.spaces.box.Box(
            low=np.array([
                -self.OBS_RELX, -self.OBS_RELY,
                0.0,
                -self.OBS_OBSX, -self.OBS_OBSY,
                0.0,
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

        self.scenario = scenario

        self.prev_yaw       = 0.0
        self.prev_roll      = 0.0
        self.prev_pitch     = 0.0
        self.prev_yaw_delta = 0.0

        self.checkpoints            = []
        self.checkpoint_objects     = []
        self.current_checkpoint_idx = 0

        self.obstacle_objects   = []
        self.obstacle_positions = []
        self.obstacle_homes     = []
        self.ramp_objects       = []
        self.ramps_taken        = set()

        self.has_obstacle  = False
        self.obstacle_pos  = None
        self.goal_object   = None
        self.goal          = None

    # ── Reset ──────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super(SimpleDrivingEnv, self).reset(seed=seed)

        self._p.resetSimulation()
        self._p.setTimeStep(self._timeStep)
        self._p.setGravity(0, 0, -10)
        Plane(self._p)
        self.car = Car(self._p)

        # All circuit scenarios spawn the car in the S/F zone (between CP1
        # and CP8) facing -x rather than at the URDF default origin.
        spawn_yaw = math.pi
        orn = self._p.getQuaternionFromEuler([0.0, 0.0, spawn_yaw])
        self._p.resetBasePositionAndOrientation(
            self.car.car, [6.0, -14.0, 0.1], orn,
        )

        self._envStepCounter = 0
        self.done            = False
        self.reached_goal    = False

        self.obstacle_objects   = []
        self.obstacle_positions = []
        self.obstacle_homes     = []
        self.ramp_objects       = []
        self.ramps_taken        = set()
        self.has_obstacle       = False
        self.obstacle_pos       = None

        self._obstacle_source, self._ramp_source, self._checkpoint_source = \
            self._sources_for_scenario(self.scenario)

        if self.scenario in self._SCENARIOS_WITH_OBSTACLES:
            self._spawn_obstacles()
        if self.scenario in self._SCENARIOS_WITH_RAMPS:
            self._spawn_ramps()

        checkpoints = options.get("checkpoints", None) if options else None
        self.checkpoints = checkpoints if checkpoints is not None else self._checkpoint_source
        self.checkpoint_objects = [Goal(self._p, pos) for pos in self.checkpoints]
        self.current_checkpoint_idx = 0

        self.goal_object = self.checkpoint_objects[0]
        self.goal        = self.checkpoints[0]

        car_pos = self.car.get_observation()
        self.prev_dist_to_goal = math.hypot(
            car_pos[0] - self.checkpoints[0][0],
            car_pos[1] - self.checkpoints[0][1],
        )
        _, car_orn = self._p.getBasePositionAndOrientation(self.car.car)
        euler = self._p.getEulerFromQuaternion(car_orn)
        self.prev_pitch, self.prev_roll, self.prev_yaw = euler
        self.prev_yaw_delta = 0.0

        return np.array(self.getExtendedObservation(), dtype=np.float32), {}

    def _sources_for_scenario(self, scenario):
        """Return (obstacle_source, ramp_source, checkpoint_source) for a scenario.
        Single source of truth — fail fast on unknown scenario names."""
        if scenario == "circuit_easy":
            return [], [], self.CIRCUIT_CHECKPOINTS
        if scenario == "circuit_medium":
            return [], self.CIRCUIT_RAMP_POSITIONS, self.CIRCUIT_CHECKPOINTS
        if scenario == "circuit_hard":
            return (self.CIRCUIT_OBSTACLE_HOMES[:self.CIRCUIT_HARD_OBSTACLE_COUNT],
                    self.CIRCUIT_RAMP_POSITIONS,
                    self.CIRCUIT_CHECKPOINTS)
        if scenario == "circuit_difficult":
            return (self.CIRCUIT_OBSTACLE_HOMES,
                    self.CIRCUIT_RAMP_POSITIONS,
                    self.CIRCUIT_CHECKPOINTS)
        raise ValueError(f"Unknown scenario: {scenario!r}")

    # ── Spawn helpers ──────────────────────────────────────────────────

    def _spawn_obstacles(self):
        self.has_obstacle       = True
        self.obstacle_homes     = list(self._obstacle_source)
        self.obstacle_positions = [self._jitter_home(h) for h in self.obstacle_homes]

        for (px, py) in self.obstacle_positions:
            vis = self._p.createVisualShape(
                shapeType=p.GEOM_CYLINDER,
                radius=0.5, length=1.0,
                rgbaColor=[1.0, 0.5, 0.0, 1.0],
            )
            body = self._p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=vis,
                basePosition=[px, py, 0.5],
            )
            self.obstacle_objects.append(body)

        self.obstacle_pos = self.obstacle_positions[0]

    def _jitter_home(self, home):
        """Sample a position uniformly in a disc of radius OBSTACLE_SPAWN_JITTER
        around `home`. Uses the env's seeded RNG so behaviour is reproducible
        per (seed, reset)."""
        r     = self.OBSTACLE_SPAWN_JITTER * math.sqrt(self.np_random.uniform(0.0, 1.0))
        theta = self.np_random.uniform(0.0, 2 * math.pi)
        return (home[0] + r * math.cos(theta), home[1] + r * math.sin(theta))

    def _spawn_ramps(self):
        for (x, y, yaw) in self._ramp_source:
            self.ramp_objects.append(Ramp(self._p, (x, y), yaw=yaw))

    # ── Step ───────────────────────────────────────────────────────────

    def step(self, action):
        if self._isDiscrete:
            fwd       = [-1, -1, -1,  0,  0,  0,  1,  1,  1]
            steerings = [-0.6, 0, 0.6, -0.6, 0, 0.6, -0.6, 0, 0.6]
            action = [fwd[action], steerings[action]]

        self.car.apply_action(action)
        nearest_obstacle = None
        collision = False

        for _ in range(self._actionRepeat):
            self._p.stepSimulation()
            if self._renders:
                time.sleep(self._timeStep)

            if self.obstacle_objects:
                car_pos, _ = self._p.getBasePositionAndOrientation(self.car.car)
                nearest_obstacle, nearest_dist = self._nearest_obstacle(car_pos)
                if nearest_dist < self.OBSTACLE_COLLISION_RADIUS:
                    collision = True
                    break

        # Obstacles are static within an episode — per-reset spawn jitter
        # is applied in _spawn_obstacles, no intra-episode drift.

        if collision:
            self.done = True

        self._envStepCounter += 1

        if self._termination():
            self.done = True

        car_pos, car_orn = self._p.getBasePositionAndOrientation(self.car.car)
        pitch, roll, yaw = self._p.getEulerFromQuaternion(car_orn)
        current_goal     = self.checkpoints[self.current_checkpoint_idx]

        # Ramp-taken detection (fires once per ramp per episode).
        ramp_taken_idx = self._check_ramp_taken(car_pos, pitch)

        dist_to_goal = math.hypot(
            car_pos[0] - current_goal[0],
            car_pos[1] - current_goal[1],
        )

        self.reached_goal = False
        if dist_to_goal < self.CHECKPOINT_RADIUS:
            self.reached_goal = True
            self.current_checkpoint_idx += 1

            if self.current_checkpoint_idx >= len(self.checkpoints):
                self.done = True
            else:
                next_goal    = self.checkpoints[self.current_checkpoint_idx]
                dist_to_goal = math.hypot(
                    car_pos[0] - next_goal[0], car_pos[1] - next_goal[1],
                )
                self.prev_dist_to_goal = dist_to_goal
                current_goal = next_goal
                self.goal_object = self.checkpoint_objects[self.current_checkpoint_idx]
                self.goal        = current_goal

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
            prev_yaw=self.prev_yaw,         current_yaw=yaw,
            prev_yaw_delta=self.prev_yaw_delta,
            prev_roll=self.prev_roll,       current_roll=roll,
            prev_pitch=self.prev_pitch,     current_pitch=pitch,
            obstacle_positions=self.obstacle_positions,
            scenario=self.scenario,
            ramp_taken=ramp_taken_idx is not None,
        )

        self.prev_yaw_delta = _wrap_delta(yaw - self.prev_yaw)
        self.prev_yaw   = yaw
        self.prev_roll  = roll
        self.prev_pitch = pitch
        self.prev_dist_to_goal = dist_to_goal

        return np.array(self.getExtendedObservation(), dtype=np.float32), float(reward), self.done, False, {}

    # ── Obstacle helpers ───────────────────────────────────────────────

    def _nearest_obstacle(self, car_pos):
        best = (None, float("inf"))
        for pos in self.obstacle_positions:
            d = math.hypot(car_pos[0] - pos[0], car_pos[1] - pos[1])
            if d < best[1]:
                best = (pos, d)
        return best

    # ── Ramp-taken detection ──────────────────────────────────────────

    def _check_ramp_taken(self, car_pos, pitch):
        if abs(pitch) < self.RAMP_PITCH_THRESH:
            return None
        for i, (rx, ry, _) in enumerate(self._ramp_source):
            if i in self.ramps_taken:
                continue
            if math.hypot(car_pos[0] - rx, car_pos[1] - ry) < self.RAMP_TAKEN_RADIUS:
                self.ramps_taken.add(i)
                print(f"RAMP {i} TAKEN at pitch={pitch:.2f}, pos=({car_pos[0]:.1f},{car_pos[1]:.1f})")
                return i
        return None

    # ── Camera ─────────────────────────────────────────────────────────

    def get_camera_image(self, width=None, height=None):
        """
        Render an 84x84 RGB frame from a car-mounted, forward-looking camera.

        The camera sits slightly above and ahead of the car origin and looks
        along the car's current heading (yaw), so the view rotates with the
        car — unlike a fixed world-frame camera. Returns a uint8 array of
        shape (H, W, 3).

        This method is purely additive: it does not touch step/reset/reward
        or the observation vector, so it cannot affect existing training.
        It is called externally (data collection, vision inference).
        """
        width  = width  if width  is not None else self.CAM_WIDTH
        height = height if height is not None else self.CAM_HEIGHT

        car_pos, car_orn = self._p.getBasePositionAndOrientation(self.car.car)
        _, _, yaw = self._p.getEulerFromQuaternion(car_orn)

        cos_y, sin_y = math.cos(yaw), math.sin(yaw)

        cam_eye = [
            car_pos[0] + self.CAM_EYE_FWD * cos_y,
            car_pos[1] + self.CAM_EYE_FWD * sin_y,
            car_pos[2] + self.CAM_EYE_HEIGHT,
        ]
        cam_target = [
            car_pos[0] + self.CAM_TARGET_FWD * cos_y,
            car_pos[1] + self.CAM_TARGET_FWD * sin_y,
            car_pos[2] - 0.1,    # was +0.1 — now looks slightly downward
        ]

        view = self._p.computeViewMatrix(
            cameraEyePosition=cam_eye,
            cameraTargetPosition=cam_target,
            cameraUpVector=[0, 0, 1],
        )
        proj = self._p.computeProjectionMatrixFOV(
            fov=self.CAM_FOV,
            aspect=float(width) / height,
            nearVal=self.CAM_NEAR,
            farVal=self.CAM_FAR,
        )
        _, _, px, _, _ = self._p.getCameraImage(
            width=width, height=height,
            viewMatrix=view, projectionMatrix=proj,
        )
        return np.array(px, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]

    # ── Observation ────────────────────────────────────────────────────

    def getExtendedObservation(self):
        car_pos, car_orn = self._p.getBasePositionAndOrientation(self.car.car)
        pitch, roll, yaw = self._p.getEulerFromQuaternion(car_orn)
        vel   = self._p.getBaseVelocity(self.car.car)[0]
        speed = math.hypot(vel[0], vel[1])

        idx = min(self.current_checkpoint_idx, len(self.checkpoints) - 1)
        cur = self.checkpoints[idx]
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
        car_pos, car_orn = self._p.getBasePositionAndOrientation(self.car.car)
        pitch, roll, _ = self._p.getEulerFromQuaternion(car_orn)
        lap_complete  = self.current_checkpoint_idx >= len(self.checkpoints)
        out_of_bounds = abs(car_pos[0]) > 50 or abs(car_pos[1]) > 30
        flipped       = abs(pitch) > self.FLIP_THRESHOLD or abs(roll) > self.FLIP_THRESHOLD
        return self._envStepCounter > 500 or lap_complete or out_of_bounds or flipped