import math
import pybullet as p


class Ramp:
    """
    A wedge-shaped ramp the car can drive up.

    Built as a triangular prism (not a tilted box): the thin leading edge
    sits flush with the ground so the car rolls onto the slope, which rises
    over `length` to `height` at the back. Extruded by `width` along its
    local y-axis. `yaw` rotates which compass direction the slope faces.

    With the defaults (length=3.0, height=0.4) the slope is ~7.6 degrees —
    gentle enough to drive up, steep enough to launch the car at speed.
    """

    def __init__(self, client, base_position, yaw=0.0,
                 length=3.0, width=2.0, height=0.4):
        self.client = client
        self.position = base_position
        self.yaw = yaw

        L = length / 2.0
        W = width / 2.0

        # Local frame: low edge (z=0) at -x, apex (z=height) at +x.
        # Car approaches from the -x side and drives up the hypotenuse.
        vertices = [
            [-L, -W, 0.0],     # 0 front-bottom, -y
            [ L, -W, 0.0],     # 1 back-bottom,  -y
            [ L, -W, height],  # 2 back-top,     -y
            [-L,  W, 0.0],     # 3 front-bottom, +y
            [ L,  W, 0.0],     # 4 back-bottom,  +y
            [ L,  W, height],  # 5 back-top,     +y
        ]
        # 8 triangles: two end caps, slope face, bottom face, back face.
        indices = [
            0, 1, 2,            # -y cap
            3, 5, 4,            # +y cap
            0, 2, 5,  0, 5, 3,  # slope (drivable hypotenuse)
            0, 3, 4,  0, 4, 1,  # bottom (rests on ground)
            1, 4, 5,  1, 5, 2,  # back vertical face
        ]

        # Collision: GEOM_MESH from vertices builds the convex hull, which for
        # a triangular prism is the exact shape (it is convex).
        col_shape_id = client.createCollisionShape(
            shapeType=p.GEOM_MESH,
            vertices=vertices,
        )
        vis_shape_id = client.createVisualShape(
            shapeType=p.GEOM_MESH,
            vertices=vertices,
            indices=indices,
            rgbaColor=[1.0, 0.6, 0.1, 1.0],   # orange wedge
        )

        # Bottom face is at local z=0, so placing the body at world z=0 rests
        # the ramp flush on the ground plane. Only yaw is applied.
        orn = p.getQuaternionFromEuler([0.0, 0.0, yaw])

        self.body = client.createMultiBody(
            baseMass=0,  # static
            baseCollisionShapeIndex=col_shape_id,
            baseVisualShapeIndex=vis_shape_id,
            basePosition=[base_position[0], base_position[1], 0.0],
            baseOrientation=orn,
        )

    def get_id(self):
        return self.body