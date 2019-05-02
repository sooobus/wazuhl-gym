import logging
import random
import gym

env = gym.make('wazuhl_gym:wazuhl-v0')
env.reset()
done = False
while not done:
    env.render()
    state, reward, done, _ = env.step(env.action_space.sample()) # take a random action
    logging.info("Reward: {}, State len: {}, Done: {}".format(reward, len(state), done))
env.close()