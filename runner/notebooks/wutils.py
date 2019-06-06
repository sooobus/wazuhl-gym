from collections import Counter
from collections import namedtuple
from tqdm import tqdm_notebook as tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

Record = namedtuple('Record', ['init_state', 'taken_actions', 'compile_time', 'exec_time', 'test_name'])
Batch = namedtuple('Batch', ['state', 'actions', 'state_lens', 'target'])
ModelInfo = namedtuple('ModelInfo', ['model', 'state_vocab', 'actions_vocab', 'bpe_order'])


def get_n_trajectories(env, n, length):
    records = []
    actions = get_possible_actions(env)
    for _ in tqdm(range(n)):
        init_state, reward, done, _ = env.reset()
        chosen_actions = np.random.choice(actions, length)
        for a in chosen_actions:
            state, reward, done, _ = env.step_name(a)
        state, reward, done, _ = env.step_name("terminal")
        if reward:
            records.append(Record(init_state, chosen_actions, reward[0], reward[1], env.get_name()))
        else:
            records.append(Record(init_state, chosen_actions, None, None, env.get_name()))
    return records


def get_possible_actions(env):
    possible_actions = env.action_names
    return list(set(possible_actions) - set(["terminal", "empty"]))


def construct_bpe(sequences, iterations=10):
    sequences = [[(el, ) for el in s] for s in sequences]
    bpe_order = []
    for _ in tqdm(range(iterations)):
        freqs = Counter()
        words = Counter()
        for seq in tqdm(sequences):
            for i in range(len(seq) - 1):
                words[seq[i]] += 1
                freqs[seq[i] + seq[i + 1]] += 1
        top = freqs.most_common(1)
        #print("Vocab size is: ", len(words))
        if not top:
            return sequences
        pair, _ = top[0]
        new_sequences = []
        #print(pair)
        bpe_order.append(pair)
        for seq in tqdm(sequences):
            new_seq = []
            skip = False
            for i in range(len(seq)):
                if skip:
                    skip = False
                    continue
                if i + 1 < len(seq) and seq[i] + seq[i + 1] == pair:
                    new_seq.append(pair)
                    skip = True
                else:
                    new_seq.append(seq[i])
            new_sequences.append(new_seq)
        sequences = new_sequences
    return sequences, bpe_order


def construct_bpe_from_substitutions(sequences, subs):
    sequences = [[(el, ) for el in s] for s in sequences]
    for pair in subs:
        new_sequences = []
        #print(pair)
        for seq in sequences:
            new_seq = []
            skip = False
            for i in range(len(seq)):
                if skip:
                    skip = False
                    continue
                if i + 1 < len(seq) and seq[i] + seq[i + 1] == pair:
                    new_seq.append(pair)
                    skip = True
                else:
                    new_seq.append(seq[i])
            new_sequences.append(new_seq)
        sequences = new_sequences
    return sequences


def build_state_vocab(states):
    vocab = {}
    vocab["PAD"] = 0
    cnt = 1
    for state in states:
        for word in state:
            if word not in vocab:
                vocab[word] = cnt
                cnt += 1
    return vocab


def encode_bpe_to_tokens(bpe_states, vocab):
    answer = []
    for state in bpe_states:
        answer.append([vocab[word] for word in state])
    return answer


def pad_states(states, max_len):
    res = np.zeros((len(states), max_len))
    for i in range(len(states)):
        res[i, :len(states[i])] = np.array(states[i])
    return res


def generate_batches(data, batch_size, limit=10, bpe_iterations=100, state_len_thres=800):
    data = data.dropna()
    data = data.head(limit)
    states = np.array(data.init_state)
    actions = np.array(data.taken_actions)
    encoded_states, bpe_order = construct_bpe(states, iterations=bpe_iterations)
    vocab = build_state_vocab(encoded_states)
    actions_vocab = build_state_vocab(actions)
    encoded_states = encode_bpe_to_tokens(encoded_states, vocab)
    encoded_states = [state[:state_len_thres] for state in encoded_states]
    state_lens = np.array([len(state) for state in encoded_states])
    states = pad_states(encoded_states, state_len_thres)
    actions = np.array(encode_bpe_to_tokens(actions, actions_vocab))
    targets = np.array(data.exec_time)

    n_batches = len(data) // batch_size
    res = []
    for i in range(n_batches):
        i1 = i * batch_size
        i2 = (i + 1) * batch_size
        res.append(Batch(torch.tensor(states[i1:i2], dtype=torch.long),
                         torch.tensor(actions[i1:i2], dtype=torch.long),
                         torch.tensor(state_lens[i1:i2], dtype=torch.float32),
                         torch.tensor(targets[i1:i2], dtype=torch.float32)))
    return res, (vocab, actions_vocab, bpe_order)


def prepare_state(state, bpe_order, state_vocab, state_len_thres, repeat):
    encoded_states = construct_bpe_from_substitutions([state], bpe_order)
    encoded_states = encode_bpe_to_tokens(encoded_states, state_vocab)
    encoded_states = [state[:state_len_thres] for state in encoded_states]
    state_lens = np.array([len(state) for state in encoded_states])
    states = pad_states(encoded_states, state_len_thres)
    return torch.tensor(states, dtype=torch.long).repeat(repeat, 1), torch.tensor(state_lens, dtype=torch.float32).repeat(repeat, 1)


def prepare_actions(actions, actions_vocab):
    return torch.tensor(encode_bpe_to_tokens(actions, actions_vocab), dtype=torch.long)