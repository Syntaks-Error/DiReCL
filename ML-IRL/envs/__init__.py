#!/usr/bin/env python

from gymnasium.envs.registration import register


register(
    id="ContinuousVecGridEnv-v0",
    entry_point="envs.vectorized_grid:ContinuousGridEnv",
)

register(
    id="GoalGrid-v0",
    entry_point="envs.goal_grid:GoalContinuousGrid",
)

register(
    id="ReacherDraw-v0",
    entry_point="envs.reacher_trace:ReacherTraceEnv",
)

register(id="HopperFH-v0", entry_point="envs.mujocoFH:MujocoFH", kwargs=dict(env_name="Hopper-v4"))

register(id="Walker2dFH-v0", entry_point="envs.mujocoFH:MujocoFH", kwargs=dict(env_name="Walker2d-v4"))

register(id="HalfCheetahFH-v0", entry_point="envs.mujocoFH:MujocoFH", kwargs=dict(env_name="HalfCheetah-v4"))

register(id="AntFH-v0", entry_point="envs.mujocoFH:MujocoFH", kwargs=dict(env_name="Ant-v4"))


register(
    id="PointMazeRight-v0",
    entry_point="envs.point_maze_env:PointMazeEnv",
    kwargs={"sparse_reward": False, "direction": 1},
)
register(
    id="PointMazeLeft-v0",
    entry_point="envs.point_maze_env:PointMazeEnv",
    kwargs={"sparse_reward": False, "direction": 0},
)

# A modified ant which flips over less and learns faster via TRPO
register(id="CustomAnt-v0", entry_point="envs.ant_env:CustomAntEnv", kwargs={"gear": 30, "disabled": False})
register(id="DisabledAnt-v0", entry_point="envs.ant_env:CustomAntEnv", kwargs={"gear": 30, "disabled": True})
