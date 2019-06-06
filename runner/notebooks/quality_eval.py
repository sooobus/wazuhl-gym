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
        init_state, reward, done, _ = self.env.reset()
        a = model(init_state)
        steps = 0
        reward = None
        while a != "terminal" and steps < actions_limit:
            state, reward, done, _ = self.env.step_name(a)
            steps += 1
            a = model(state)
        state, reward, done, _ = self.env.step_name("terminal")
        return reward

    def _compare_rewards(self, rewards):
        print(rewards)

    def compare_models(self, run_number=100, actions_limit=float("inf")):
        rewards = defaultdict(list)
        for _ in tqdm(range(run_number)):
            for model in self.models:
                rewards[str(model)].append(self._run_model(model, actions_limit))
        self._compare_rewards(rewards)