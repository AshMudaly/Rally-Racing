import math
import pybullet as p


class Ramp:
    """
    A wedge-shaped ramp the car can drive over to take a shortcut/jump.
    Hitting it launches the car briefly into the air. The reward function
    can encourage this (faster lap) or discourage it (chassis instability)
    depending on weights.
    """

    def __init__(self, client, base_position, yaw=0.0,
                 length=2.0, width=2.0, height=0.4):
        self.client = client
        self.position = base_position
        self.yaw = yaw

        # Build the wedge from a half-box collision shape rotated about its y-axis.
        # Simpler approach: a thin sloped box, with collision and visual at a tilt.
        # We approximate with a box tilted so its top face is a ramp.
        half_extents = [length / 2.0, width / 2.0, height / 2.0]
        col_shape_id = client.createCollisionShape(
            shapeType=p.GEOM_BOX, halfExtents=half_extents,
        )
        vis_shape_id = client.createVisualShape(
            shapeType=p.GEOM_BOX, halfExtents=half_extents,
            rgbaColor=[1.0, 0.6, 0.1, 1.0],   # orange wedge
        )

        # Tilt the box around its local y-axis so it forms a ramp.
        # The pitch angle equals atan2(height, length).
        pitch = math.atan2(height, length)
        # Combine yaw (around world Z) with pitch
        orn = p.getQuaternionFromEuler([0.0, pitch, yaw])

        # Place so the bottom edge sits on the ground
        z = (height / 2.0) * math.cos(pitch)

        self.body = client.createMultiBody(
            baseMass=0,  # static
            baseCollisionShapeIndex=col_shape_id,
            baseVisualShapeIndex=vis_shape_id,
            basePosition=[base_position[0], base_position[1], z],
            baseOrientation=orn,
        )

    def get_id(self):
        return self.body
