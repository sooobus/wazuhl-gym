import numpy as np
from collections import deque
from tqdm import tqdm_notebook as tqdm
from wutils import prepare_actions, prepare_state
from itertools import combinations
import random

class BaseComparableModel:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name


class RandomActionsModel(BaseComparableModel):
    def __init__(self, possible_actions, seq_len=45):
        super(RandomActionsModel, self).__init__("Random Actions with len={}".format(seq_len))
        self.len = seq_len
        self.possible_actions = possible_actions
        self.call_cnt = 0

    def __call__(self, state):
        self.call_cnt += 1
        if self.call_cnt == self.len:
            self.call_cnt = 0
            return "terminal"
        else:
            return np.random.choice(self.possible_actions)


class SupervisedActionsGenerator(BaseComparableModel):
    """ Generates action sequence given initial state """
    def __init__(self, action_state_estimator, possible_actions,
                 state_vocab, actions_vocab, bpe_order, batch_size, model_action_len=3, pieces=15, state_len_thres=800):
        super(SupervisedActionsGenerator, self).__init__("Supervised Actions with len={} and pieces={}".format(model_action_len,
                                                                                                               pieces))
        self.model = action_state_estimator
        self.possible_actions = possible_actions
        self.state_vocab = state_vocab
        self.actions_vocab = actions_vocab
        self.bpe_order = bpe_order
        self.model_action_len = model_action_len
        self.pieces = pieces
        self.state_len_thres = state_len_thres
        self.batch_size = batch_size
        self.left_actions = deque()

    def _choose_best_actions(self, state):
        possible_combs = list(combinations(self.possible_actions, self.model_action_len))
        random.shuffle(possible_combs)
        possible_combs = np.array(possible_combs)
        possible_combs = possible_combs[:3200]
        total_size = len(possible_combs)
        model_inp_state, model_inp_lengths = prepare_state(state, self.bpe_order, self.state_vocab,
                                                           self.state_len_thres, repeat=self.batch_size)
        all_estimates = []
        for i in tqdm(range(total_size // self.batch_size)):
            i1 = i * self.batch_size
            i2 = (i + 1) * self.batch_size
            model_inp_actions = prepare_actions(possible_combs[i1:i2], self.actions_vocab)
            estimates = self.model(model_inp_state, model_inp_actions, model_inp_lengths.view(self.batch_size, 1))
            all_estimates.append(estimates.detach().numpy())
        all_estimates = np.concatenate(all_estimates)
        idx = np.argpartition(all_estimates, self.pieces)
        return np.append(np.concatenate(possible_combs[idx[:self.pieces]]), "terminal")

    def __call__(self, state):
        if not self.left_actions:
            best_actions = self._choose_best_actions(state)
            self.left_actions.extend(best_actions)
        next_action = self.left_actions.popleft()
        return next_action