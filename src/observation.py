"""
Observation callback for SimpleDrivingEnv.

RallyDrivingEnv builds its own 11-dim observation in `getExtendedObservation`,
so this module is only used when running the base SimpleDrivingEnv (quiz2-style
single-goal scenarios). Kept separate so the logic can be unit-tested.
"""


def custom_observation(client, car_pos, car_orn, goal_pos, goal_orn,
                       obstacle_pos, has_obstacle):
    """
    Convert global positions into the car's local frame.

    Returns: [goal_x_local, goal_y_local, obs_x_local, obs_y_local, has_obstacle]
    """
    local_car_pos, local_car_orn = client.invertTransform(car_pos, car_orn)
    local_goal_pos, _ = client.multiplyTransforms(
        local_car_pos, local_car_orn, goal_pos, goal_orn,
    )

    if has_obstacle:
        obstacle_3d = (obstacle_pos[0], obstacle_pos[1], 0.0)
        local_obs_pos, _ = client.multiplyTransforms(
            local_car_pos, local_car_orn, obstacle_3d, (0, 0, 0, 1),
        )
    else:
        local_obs_pos = (0.0, 0.0, 0.0)

    return [
        local_goal_pos[0], local_goal_pos[1],
        local_obs_pos[0], local_obs_pos[1],
        float(has_obstacle),
    ]
