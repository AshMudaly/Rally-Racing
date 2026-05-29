"""
Vision-based rally driving environment.

Replaces the privileged obstacle channels in the observation with
predictions from a frozen CNN that takes the car's onboard camera frame.

Observation slot mapping (unchanged from RallyDrivingEnv):
    [0] local_goal_x          — privileged (waypoint from track)
    [1] local_goal_y          — privileged (waypoint from track)
    [2] checkpoints_remaining — privileged
    [3] local_obs_x           — REPLACED: CNN predicted x in metres
    [4] local_obs_y           — REPLACED: CNN predicted y in metres
    [5] has_obstacle          — REPLACED: CNN visibility (0 or 1 after threshold)
    [6] pitch                 — privileged (proprioception)
    [7] roll                  — privileged (proprioception)
    [8] speed                 — privileged (proprioception)
    [9] cos(yaw)              — privileged (proprioception)
    [10] sin(yaw)             — privileged (proprioception)

Slots 0-2 stay privileged: the marker said a real rally car gets waypoints
from a route map, not from recognising which cone is "next". Slots 6-10 are
proprioceptive — no real car uses a camera for its own speed/orientation.
Only the obstacle channel is genuinely a perception problem, so only that
channel is replaced by vision.

Physics and reward are UNCHANGED: collision detection still uses true
obstacle positions internally. We are only replacing what the POLICY sees,
not what the world does. This keeps the comparison clean.
"""

import numpy as np
import torch

from simple_driving.envs.rally_driving_env import RallyDrivingEnv


class VisionRallyDrivingEnv(RallyDrivingEnv):
    """
    RallyDrivingEnv with obstacle channels replaced by a frozen CNN.

    The model is passed in (not loaded internally) so the caller controls
    where weights live, and so multiple env instances can share one model
    without re-loading.
    """

    def __init__(self,
                 vision_model=None,
                 vis_threshold: float = 0.5,
                 **kwargs):
        super().__init__(**kwargs)
        if vision_model is None:
            raise ValueError(
                "VisionRallyDrivingEnv requires `vision_model` — pass a "
                "loaded ObstacleCNN instance via gym.make(..., vision_model=cnn)."
            )
        self.vision_model  = vision_model
        self.vis_threshold = vis_threshold
        self.vision_model.eval()   # defensive — predict() also does this

        # Diagnostic counters, populated each step. Useful for the report:
        # "the CNN voted 'obstacle visible' in N% of frames."
        self.last_vis_prob = 0.0
        self.last_pred_xy  = (0.0, 0.0)

    def getExtendedObservation(self):
        # Compute privileged obs first; we then OVERWRITE slots 3,4,5.
        obs = super().getExtendedObservation()

        # Capture the current camera frame (NHWC uint8 -> NCHW uint8 batch).
        img = self.get_camera_image()                    # (84, 84, 3) uint8
        img_t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # (1, 3, 84, 84) uint8

        vis_prob, pos_m = self.vision_model.predict(img_t)
        vis_prob = float(vis_prob.item())
        px, py   = float(pos_m[0, 0].item()), float(pos_m[0, 1].item())

        self.last_vis_prob = vis_prob
        self.last_pred_xy  = (px, py)

        # Splice in CNN outputs. When the CNN doesn't see an obstacle, zero
        # out the position channels — matches what the policy was trained
        # to see in phase1 (has_obstacle=0 with local_obs_pos=(0,0)).
        if vis_prob > self.vis_threshold:
            obs[3] = px
            obs[4] = py
            obs[5] = 1.0
        else:
            obs[3] = 0.0
            obs[4] = 0.0
            obs[5] = 0.0

        return obs