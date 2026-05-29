"""
Obstacle-perception CNN.

Takes an 84x84 RGB frame from the car's camera and outputs:
    vis_logit : scalar, logit of "obstacle visible in frame" (sigmoid -> prob)
    pos       : (x, y) in CAR-LOCAL frame, in METRES, of the nearest visible
                obstacle. Only meaningful when sigmoid(vis_logit) > 0.5.

Architecture: three stride-2 convs (down to 11x11) + a 128-dim trunk + two
linear heads. ~1M params, trains in minutes on CPU.

Position labels are normalised by POS_SCALE during training so the regression
loss is on the same order as BCE. The model output is in normalised space;
`predict()` undoes the scaling so callers get metres directly.
"""

import torch
import torch.nn as nn

POS_SCALE = 10.0   # metres. labels/outputs are divided/multiplied by this.


class ObstacleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 84 -> 42 -> 21 -> 11
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.trunk = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 11 * 11, 128),
            nn.ReLU(inplace=True),
        )
        self.vis_head = nn.Linear(128, 1)
        self.pos_head = nn.Linear(128, 2)

    def forward(self, x: torch.Tensor):
        """
        x: (B, 3, 84, 84) float in [0,1] OR uint8 in [0,255].
        Returns (vis_logit: (B,1), pos_norm: (B,2)) — pos is normalised.
        """
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        f = self.trunk(self.conv(x))
        return self.vis_head(f), self.pos_head(f)

    @torch.no_grad()
    def predict(self, x: torch.Tensor):
        """
        Inference helper. Returns (visible_prob: (B,), pos_metres: (B,2)).
        Use this from the env / interface layer, not forward().
        """
        self.eval()
        vis_logit, pos_norm = self.forward(x)
        vis_prob = torch.sigmoid(vis_logit).squeeze(-1)
        pos_m    = pos_norm * POS_SCALE
        return vis_prob, pos_m