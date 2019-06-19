from o2actions import o2actions

class DropKeepModel:
    def __init__(self, env):
        self.env = env
        self.o2seq = o2actions
        self.step_counter = 0

    def step(self, skip):
        action = "empty"
        if not skip:
            action = self.o2seq[self.step_counter]
        self.step_counter += 1
        new_state, reward, done, _ = self.env.step(action)
        if done:
            self.step_counter = 0
        return new_state, reward, done, _