from collections import Counter
from collections import namedtuple
from tqdm import tqdm_notebook as tqdm

Record = namedtuple('Record', ['init_state', 'taken_actions', 'compile_time', 'exec_time', 'test_name'])

def get_n_trajectories(env, n, length):
    records = []
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
        for seq in sequences:
            for i in range(len(seq) - 1):
                words[seq[i]] += 1
                freqs[seq[i] + seq[i + 1]] += 1
        top = freqs.most_common(1)
        print("Vocab size is: ", len(words))
        if not top:
            return sequences
        pair, _ = top[0]
        new_sequences = []
        print(pair)
        bpe_order.append(pair)
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
    return sequences, bpe_order


def construct_bpe_from_substitutions(sequences, subs):
    sequences = [[(el, ) for el in s] for s in sequences]
    for pair in subs:
        new_sequences = []
        print(pair)
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