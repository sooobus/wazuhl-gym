import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding
from wazuhl_gym.envs.interactor import Interactor
import logging

class WazuhlEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.tests = 1
        self.interactor = Interactor()
        self.action_names = self.interactor.get_possible_actions()
        self.action_space = spaces.Discrete(len(self.action_names))
        self.action_to_index = {self.action_names[i]: i for i in range(len(self.action_names))}

    def step(self, action_):
        action = self.action_names[action_]
        self.interactor.send_action(action)
        logging.info("Sent action {}".format(action))
        idx, state, done = self.interactor.get_state()
        logging.info("State len: {}".format(len(state)))
        logging.info("Done: {}".format(done))
        reward = 0
        if done:
            rewards = self.interactor.get_rewards()
            reward = self.calc_reward(*rewards)
        return state, reward, done, {}

    def calc_reward(self, compile_time, exec_time):
        return -(compile_time + exec_time)

    def reset(self):
        pass

    def render(self, mode='human'):
        pass

    def close(self):
        pass