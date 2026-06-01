from gymnasium.envs.registration import register

register(
    id='SimpleDriving-v0',
    entry_point='simple_driving.envs:SimpleDrivingEnv',
    max_episode_steps=None,
)

register(
    id='RallyDriving-v0',
    entry_point='simple_driving.envs:RallyDrivingEnv',
    max_episode_steps=None,
)

register(
    id='VisionRallyDriving-v0',
    entry_point='simple_driving.envs:VisionRallyDrivingEnv',
    max_episode_steps=None,
)
