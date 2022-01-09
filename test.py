# import gym
# from gym import wrappers
# import highway_env
# from stable_baselines3 import DQN

# env = gym.make("highway-fast-v0")
# model = DQN('MlpPolicy', env,
#               policy_kwargs=dict(net_arch=[256, 256]),
#               learning_rate=5e-4,
#               buffer_size=15000,
#               learning_starts=200,
#               batch_size=32,
#               gamma=0.8,
#               train_freq=1,
#               gradient_steps=1,
#               target_update_interval=50,
#               verbose=1,
#               tensorboard_log="highway_dqn/")
# model.learn(int(2e4))
# model.save("highway_dqn/model")

# # Load and test saved model
# model = DQN.load("highway_dqn/model")
# while True:
#   done = False
#   obs = env.reset()
#   while not done:
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
#     env.render()

import gym
# import pprint
import safety_gym
import highway_env
import matplotlib.pyplot as plt

env = gym.make('highway-v0')
# env = gym.make("merge-v0")
env.reset()
collide = False
for _ in range(200):
    # action = env.action_type.actions_indexes["IDLE"]
	# if not collide:
	action = [1, 0.0]
	# else:
	# 	action = env.action_space.sample()
	# print(action)
	
	obs, reward, done, info = env.step(action)
	collide = info['crashed']
	cost = info['cost']
	print(f'cost = {cost}, reward = {reward}')
	# print(info['cost'])
	env.render()

# plt.imshow(env.render(mode="rgb_array"))
# plt.show()


# # env1 = gym.make('Safexp-CarGoal1-v0')
# env2 = gym.make('highway-v0')

# # print(env1.action_space)
# print(env2.action_space)
# pprint.pprint(env2.config)
# # pprint.pprint(env1.config)

# i = 0
# while(True):
# 	# env1.reset()
# 	env2.reset()
# 	# action1 = env1.action_space.sample()
# 	action2 = env2.action_space.sample()
# 	print('=='*20)
# 	# print(action1.shape)
# 	print(action2)
# 	# action2 = [1.0, 0.0]
# 	# action2 = [0.1, 0.1]
# 	action2 = [0.0, 0.0]
# 	# obs1, reward1, done1, info1 = env1.step(action1)
# 	obs2, reward2, done2, info2 = env2.step(action2)
# 	# print('=='*20)
# 	# print('env1 = \n')
# 	# print('obs = ', obs1)
# 	# print('reward = ', reward1)
# 	# print('done = ', done1)
# 	# print('info = ', info1)
# 	# print('env2 = \n')
# 	# print('obs = ', obs2)
# 	# print('reward = ', reward2)
# 	# print('done = ', done2)
# 	# print('info = ', info2)
# 	env2.render()
# 	i += 1
