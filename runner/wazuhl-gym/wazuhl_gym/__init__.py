from gym.envs.registration import register

register(
    id='wazuhl-v0',
    entry_point='wazuhl_gym.envs:WazuhlEnv',
)