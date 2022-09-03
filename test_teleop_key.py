# import os
# import sys
# import subprocess

# launchfile = "limo_ackerman.launch"
# command = "roslaunch limo_gazebo_sim " + launchfile
# _roslaunch = subprocess.Popen(command.split())


import time
from turtle import done
import gym
import gym_gazebo
from gym_gazebo.envs.nav_env import GazeboCarNavEnv
import numpy as np

DEFAULT_CONFIG = dict(
    action_repeat=2,
    max_episode_length=1000,
    lidar_dim=16,  # total: 450
    use_dist_reward=False,
    stack_obs=False,
    reward_distance=1.0,  # reward scale
    placements_margin=1.2,  # min distance of obstacles between each other
    goal_region=0.3,
    collision_region=0.45,
    cost_region=0.6,
    lidar_max_range=5.0,
    lvel_lim=0.3,  # linear velocity limit
    rvel_lim=0.8,  # rotational velocity limit

    # grid map
    use_grid_map=False,
    xmax=4,
    ymax=4,
    resolution=0.1,
    render=False
)


env = GazeboCarNavEnv(level=1, seed=10, config=DEFAULT_CONFIG)
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
    done = False
    env.reset()
    step_num = 0
    while not done:
        # action = [0.0, 0.0]
        linear_vel = np.random.uniform(-1, 1)
        rot_vel = np.random.uniform(-1, 1)
        action = [linear_vel, rot_vel]
        # env.step(action)
        obs, reward, done, info = env.step(action)
        step_num += 1
        # print(f"reward = {reward}, cost = {info['cost']}")
        # env.set_robot_pose()
        time.sleep(0.1)
    print(step_num)
# env.step([0.0, 0.0])
# env.close()

env.reset()