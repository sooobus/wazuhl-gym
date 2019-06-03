import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding
from wazuhl_gym.envs.interactor import Interactor
import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)

class WazuhlEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.tests = 1
        self.interactor = Interactor()
        self.action_names = self.interactor.get_possible_actions()
        self.action_space = spaces.Discrete(len(self.action_names))
        self.action_to_index = {self.action_names[i]: i for i in range(len(self.action_names))}
        self.last_compile_time = None
        self.last_exec_time = None
        self.last_step_terminal = True

    def step(self, action_):
        self.last_step_terminal = False
        action = self.action_names[action_]
        self.interactor.send_action(action)
        idx, state, done = self.interactor.get_state()
        logging.info("State len: {}".format(len(state)))
        logging.info("Done: {}".format(done))
        reward = (0, 0)
        if done:
            self.last_step_terminal = True
            rewards = self.interactor.get_rewards()
            reward = self.calc_reward(*rewards)
        return state, reward, done, {}

    def step_name(self, action):
        return self.step(self.action_to_index[action])

    def calc_reward(self, compile_time, exec_time):
        self.last_compile_time = compile_time
        self.last_exec_time = exec_time
        if not compile_time and not exec_time:
            return None
        return (compile_time, exec_time)

    def get_last_compile_exec_time(self):
        return self.last_compile_time, self.last_exec_time

    def get_name(self):
        return self.interactor.get_name()

    def reset(self):
        if not self.last_step_terminal:
            self.step_name("terminal")
        return self.step_name("empty")

    def render(self, mode='human'):
        pass

    def close(self):
        pass