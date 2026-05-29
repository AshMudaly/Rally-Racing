"""
Verify the car-mounted, yaw-following camera.

Renders frames at several car headings and saves a contact sheet so you can
confirm the view rotates WITH the car (a fixed world camera would show an
identical scene at every heading).

Run from src/:
    python3 capture_test.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

import math
import numpy as np
import pybullet as p
import gymnasium as gym
import simple_driving  # registers RallyDriving-v0
from PIL import Image

from reward import custom_reward


def main():
    env = gym.make("RallyDriving-v0", renders=False, isDiscrete=False,
                   reward_callback=custom_reward, scenario="phase3")
    env.reset()
    raw = env.unwrapped

    # Place the car near the cluster of obstacles so something is in view,
    # then rotate it through several headings and capture each.
    headings = [0.0, math.pi / 2, math.pi, -math.pi / 2]
    frames = []
    for yaw in headings:
        orn = p.getQuaternionFromEuler([0, 0, yaw])
        # Sit the car at origin-ish, looking different directions
        raw._p.resetBasePositionAndOrientation(raw.car.car, [2, 2, 0.1], orn)
        raw._p.stepSimulation()
        img = raw.get_camera_image()
        frames.append(img)
        print(f"yaw={math.degrees(yaw):+6.1f}°  frame shape={img.shape}  "
              f"mean RGB={img.reshape(-1,3).mean(0).round(1)}")

    # Stitch into a single 2x2 contact sheet (84x84 each -> 168x168)
    top    = np.hstack([frames[0], frames[1]])
    bottom = np.hstack([frames[2], frames[3]])
    sheet  = np.vstack([top, bottom])
    out = os.path.join(HERE, "camera_check.png")
    Image.fromarray(sheet).save(out)
    print(f"\nSaved contact sheet to {out}")
    print("Top-left=0°  Top-right=90°  Bottom-left=180°  Bottom-right=-90°")
    print("If the scene differs across the four tiles, the camera is "
          "correctly following the car's heading.")
    env.close()


if __name__ == "__main__":
    main()