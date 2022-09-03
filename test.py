# import os
# import sys
# import subprocess

# launchfile = "limo_ackerman.launch"
# command = "roslaunch limo_gazebo_sim " + launchfile
# _roslaunch = subprocess.Popen(command.split())


import time
import gym
import gym_gazebo
from gym_gazebo.envs.nav_env import GazeboCarNavEnv
import numpy as np


config = dict(
    action_repeat=5,
    max_episode_length=1000,
    lidar_dim=16,  # total: 450
    use_dist_reward=False,
    stack_obs=False,
    use_grid_map=True,
    resolution=0.05,
)


env = GazeboCarNavEnv(level=1, seed=10, config=config)
# env = gym.make("GazeboCarNav-v0")
# env.seed(0)
env.reset()

for i in range(10):
    # x = np.random.uniform(-0.3, 0.3)
    # z = np.random.uniform(-0.3, 0.3)
    # x = 0.1
    # # z = -0.3
    # z = -0.1
    # action = [x, z]
    # print('action = ', action)
    # env.step(action)
    # obs = env.reset()
    # for j in range(10):
    #     env.reset_goal_pos()
    #     time.sleep(2)
    # print(obs.shape)
    lv = np.random.uniform(-1, 1)
    rv = np.random.uniform(-1, 1)
    action = [lv, rv]
    env.step(action)
    # env.set_robot_pose()
    time.sleep(0.1)
# env.step([0.0, 0.0])
# env.close()

env.reset()