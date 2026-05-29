"""
Collect a labelled vision dataset for the CNN obstacle-detection head.

For each sample:
  1. reset the env (random scenario layout)
  2. pick a random obstacle as the "subject"
  3. teleport the car to a random pose around it
       - distance uniform in [DIST_MIN, DIST_MAX]
       - bearing uniform in [-BEARING_MAX, +BEARING_MAX] off the obstacle direction
       - yaw uniform in [-YAW_MAX, +YAW_MAX] off "looking at obstacle"
  4. render one frame
  5. compute labels in car frame + a geometric "visible" flag

Why this rather than policy rollouts:
  A trained driving policy avoids obstacles, so its frames are skewed to long
  range. We want close-range frames for the collision-avoidance task. Random
  poses give explicit control over the distance distribution.

Output:
  data/vision_dataset.npz with
      images:  (N, 84, 84, 3) uint8
      labels:  (N, 3) float32  -> [obs_local_x, obs_local_y, visible]
      dists:   (N,) float32    -> ground-truth distance, diagnostics only
      seed:    () int64
      sampler: () str           -> "random_pose_v1"

Run from src/:
    python3 ../vision/collect_data.py [--n 10000] [--seed 0]
"""

import argparse
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.abspath(os.path.join(HERE, "..", "src"))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, SRC)
sys.path.insert(0, ROOT)

import numpy as np
import pybullet as p
import gymnasium as gym
import simple_driving  # registers RallyDriving-v0

from reward import custom_reward

# ── Sampling parameters ────────────────────────────────────────────────────
DIST_MIN     = 0.5      # m, closest distance from obstacle
DIST_MAX     = 6.0      # m, furthest distance
BEARING_MAX  = math.radians(90)   # off-axis spread when picking the car's position around the obstacle
YAW_MAX      = math.radians(60)   # how far the car may face away from the obstacle (some samples won't see it)

# Frustum visibility check — must match env CAM_FOV
CAM_FOV_DEG  = 55.0
HALF_FOV     = math.radians(CAM_FOV_DEG / 2.0)
MAX_VIEW_DIST = 30.0    # beyond this, treat as "not visible" even if in-frustum


def make_env():
    """phase3 has obstacles AND ramps so we get the widest visual variety."""
    return gym.make(
        "RallyDriving-v0",
        renders=False,
        isDiscrete=False,
        reward_callback=custom_reward,
        observation_callback=None,
        scenario="phase3",
    )


def car_local_xy(car_pos, car_yaw, world_xy):
    """World (x,y) -> car-frame (x,y). +x forward, +y left."""
    dx = world_xy[0] - car_pos[0]
    dy = world_xy[1] - car_pos[1]
    cy, sy = math.cos(-car_yaw), math.sin(-car_yaw)
    return cy * dx - sy * dy, sy * dx + cy * dy


def is_visible(local_x, local_y, dist):
    """Geometric frustum check. True iff obstacle is in front, within FOV, within range."""
    if local_x <= 0:           # behind the car
        return False
    if dist > MAX_VIEW_DIST:
        return False
    bearing = math.atan2(local_y, local_x)
    return abs(bearing) <= HALF_FOV


def sample_one(env, rng):
    """Reset, teleport, capture one (image, label, dist) triple.

    Labelling: the nearest VISIBLE obstacle, not the nearest in world.
    If no obstacle is visible, this is a well-posed negative example
    (label = [0, 0, 0]). Ramps are not in obstacle_positions, so the
    CNN only ever sees cones as positive targets — it learns to
    distinguish them from ramps by shape.
    """
    env.reset()
    raw = env.unwrapped

    if not raw.obstacle_positions:
        raise RuntimeError("No obstacles in env — check scenario='phase3'.")

    # Pick a random obstacle to position the car NEAR. This is just for
    # pose sampling — it is NOT necessarily the labelled obstacle.
    subject_idx = rng.integers(0, len(raw.obstacle_positions))
    sx, sy = raw.obstacle_positions[subject_idx]

    dist_sample = rng.uniform(DIST_MIN, DIST_MAX)
    bearing     = rng.uniform(-BEARING_MAX, BEARING_MAX)
    cx = sx + dist_sample * math.cos(bearing)
    cy = sy + dist_sample * math.sin(bearing)

    yaw_to_subject = math.atan2(sy - cy, sx - cx)
    car_yaw        = yaw_to_subject + rng.uniform(-YAW_MAX, YAW_MAX)

    orn = p.getQuaternionFromEuler([0, 0, car_yaw])
    raw._p.resetBasePositionAndOrientation(raw.car.car, [cx, cy, 0.1], orn)
    raw._p.stepSimulation()

    img = raw.get_camera_image()

    # Visibility-aware labelling: find ALL visible obstacles, label the
    # nearest. If none are visible, label is a clean (0, 0, 0).
    car_pos         = (cx, cy)
    nearest_d_world = float("inf")
    visible         = []
    for op in raw.obstacle_positions:
        d = math.hypot(car_pos[0] - op[0], car_pos[1] - op[1])
        nearest_d_world = min(nearest_d_world, d)
        lx, ly = car_local_xy(car_pos, car_yaw, op)
        if is_visible(lx, ly, d):
            visible.append((d, lx, ly))

    if visible:
        d_visible, lx, ly = min(visible, key=lambda t: t[0])
        label = np.array([lx, ly, 1.0], dtype=np.float32)
    else:
        label = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    return img, label, float(nearest_d_world)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",    type=int, default=10_000, help="number of samples")
    parser.add_argument("--seed", type=int, default=0,      help="RNG seed")
    parser.add_argument("--out",  type=str,
                        default=os.path.join(ROOT, "data", "vision_dataset.npz"))
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rng = np.random.default_rng(args.seed)

    env = make_env()
    print(f"Collecting {args.n} samples (seed={args.seed}) -> {args.out}")
    print(f"  distance range: [{DIST_MIN}, {DIST_MAX}] m")
    print(f"  bearing spread: ±{math.degrees(BEARING_MAX):.0f}°")
    print(f"  yaw jitter:     ±{math.degrees(YAW_MAX):.0f}°")
    print(f"  FOV check:      {CAM_FOV_DEG}°")

    images = np.empty((args.n, 84, 84, 3), dtype=np.uint8)
    labels = np.empty((args.n, 3),         dtype=np.float32)
    dists  = np.empty((args.n,),           dtype=np.float32)

    t0 = time.time()
    for i in range(args.n):
        images[i], labels[i], dists[i] = sample_one(env, rng)
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate    = (i + 1) / elapsed
            eta     = (args.n - i - 1) / rate
            visible_frac = float(labels[: i + 1, 2].mean())
            print(f"  [{i+1:>6}/{args.n}]  {rate:5.1f} samples/s  "
                  f"visible_frac={visible_frac:.2f}  ETA={eta:5.0f}s")

    env.close()

    np.savez_compressed(
        args.out,
        images=images,
        labels=labels,
        dists=dists,
        seed=np.int64(args.seed),
        sampler=np.array("random_pose_v1"),
    )
    print(f"\nSaved {args.n} samples to {args.out}")
    print(f"  visible fraction: {labels[:, 2].mean():.3f}")
    print(f"  mean dist (visible):    "
          f"{dists[labels[:, 2] == 1.0].mean():.2f} m"
          if (labels[:, 2] == 1.0).any() else "  (no visible samples — bug!)")
    print(f"  local_x range: [{labels[:, 0].min():.2f}, {labels[:, 0].max():.2f}]")
    print(f"  local_y range: [{labels[:, 1].min():.2f}, {labels[:, 1].max():.2f}]")


if __name__ == "__main__":
    main()