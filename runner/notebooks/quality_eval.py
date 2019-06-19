from wutils import get_possible_actions
from tqdm import tqdm_notebook as tqdm
from collections import defaultdict


class QualityEvaluator:
    """ Takes different models and compares their performance on given environment """
    def __init__(self, env):
        self.env = env
        self.models = []
        self.actions = get_possible_actions(env)

    def add_model(self, model):
        self.models.append(model)

    def _run_model(self, model, actions_limit=float("inf")):
        #print("run model {}".format(str(model)))
        state, reward, done, _ = self.env.reset()
        a = model(state)
        steps = 0
        reward = None
        done = None
        print(a)
        while a != "terminal" and steps < actions_limit:
            if a != "empty":
                state, reward, done, _ = self.env.step_name(a)
            steps += 1
            print(a, reward, done)
            a = model(state)
        state, reward, done, _ = self.env.step_name("terminal")
        #print("terminal reward: {}".format(reward))
        return reward, self.env.get_name()

    def _compare_rewards(self, rewards):
        return rewards

    def compare_models(self, run_number=100, actions_limit=float("inf")):
        rewards = defaultdict(list)
        for _ in tqdm(range(run_number)):
            for model in self.models:
                rewards[str(model)].append(self._run_model(model, actions_limit))
        return self._compare_rewards(rewards)
