import sys
import torch
import gym
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pandas as pd

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions, vocab, hidden_size=256, learning_rate=3e-4):
        super(ActorCritic, self).__init__()

        self.num_actions = num_actions
        self.hid_size = hidden_size
        self.emb_size = hidden_size // 2
        self.voc_len = len(vocab)

        self.critic_embedding = nn.Embedding(
            num_embeddings=self.voc_len,
            embedding_dim=self.emb_size,
            padding_idx=0)
        self.critic_lstm = nn.LSTM(self.emb_size, self.hid_size, batch_first=True)
        self.critic_linear1 = nn.Linear(self.hid_size, self.hid_size // 2)
        self.critic_linear2 = nn.Linear(self.hid_size // 2, 1)

        self.critic_embedding = nn.Embedding(
            num_embeddings=self.voc_len,
            embedding_dim=self.emb_size,
            padding_idx=0)
        self.critic_lstm = nn.LSTM(self.emb_size, self.hid_size, batch_first=True)
        self.actor_linear1 = nn.Linear(self.hid_size, self.hid_size // 2)
        self.actor_linear2 = nn.Linear(self.hid_size // 2, num_actions)

    def forward(self, state):
        self.hidden_actor = (torch.randn(1, 1, self.hid_size), torch.randn(1, 1, self.hid_size))
        self.hidden_critic = (torch.randn(1, 1, self.hid_size), torch.randn(1, 1, self.hid_size))

        value = self.critic_embedding(state)
        value, self.hidden_critic = self.critic_lstm(value, self.hidden_critic)
        value = F.relu(self.critic_linear1(value))
        value = self.critic_linear2(value)

        policy_dist = self.actor_embedding(state)
        policy_dist, self.hidden_actor = self.actor_lstm(policy_dist, self.hidden_actor)
        policy_dist = F.relu(self.actor_linear1(state))
        policy_dist = F.softmax(self.actor_linear2(policy_dist), dim=1)

        return value, policy_dist