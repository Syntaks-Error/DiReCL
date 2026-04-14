"""
Observation reference for MuJoCo Ant (`Ant-v4` in Gymnasium):

obs = concat(
    qpos[2:],                # body/joint positions except global x, y
    qvel,                    # joint velocities
    clip(cfrc_ext, -1, 1)    # external contact forces
)

Typical dimensionality is 111 in Gymnasium's Ant-v4.

Reward function should be written as:

def compute_reward(obs, action, next_obs, info):
    ...
    return reward, {"component_name": component_value, ...}

`reward` and all component values must be scalar floats.
"""
