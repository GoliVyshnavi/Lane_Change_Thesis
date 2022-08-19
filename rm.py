import numpy as np
import random

# # reference-----https://github.com/Ilhem23/change_lane_DQN

class ReplayMemory(object):
    """Replay Memory that stores the last size=1,000,000 transitions"""

    def __init__(self, size=10000, vector_lenght=37,
                 agent_history_length=4, batch_size=32):
        self.size = size
        self.vector_lenght = vector_lenght
        self.agent_history_length = agent_history_length
        self.batch_size = batch_size
        self.count = 0
        self.current = 0

        # Pre-allocate memory
        self.actions = np.empty(self.size, dtype=np.int32)
        self.rewards = np.empty(self.size, dtype=np.float32)
        self.frames = np.empty((self.size, self.vector_lenght), dtype=np.uint8)
        self.terminal_flags = np.empty(self.size, dtype=np.bool)

        # Pre-allocate memory for the states and new_states in a minibatch
        self.states = np.empty((self.batch_size, self.vector_lenght), dtype=np.float32)
        self.new_states = np.empty((self.batch_size, self.vector_lenght), dtype=np.float32)
        self.indices = np.empty(self.batch_size, dtype=np.int32)

    def add_experience(self, action, frame, reward, terminal):
        if frame.shape != (self.vector_lenght, ):
            raise ValueError('Dimension of frame is wrong!')
        self.actions[self.current] = action
        self.frames[self.current, ...] = frame
        self.rewards[self.current] = reward
        self.terminal_flags[self.current] = terminal
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.size

    def _get_state(self, index):
        if self.count is 0:
            raise ValueError("empty replay memory")
        if index < self.agent_history_length - 1:
            raise ValueError("Index must be min 3")
        return self.frames[index - self.agent_history_length + 1:index + 1, ...]

    def _get_valid_indices(self):
        for i in range(self.batch_size):
            while True:
                index = random.randint(self.agent_history_length, self.count - 1)
                if index < self.agent_history_length:
                    continue
                if index >= self.current and index - self.agent_history_length <= self.current:
                    continue
                if self.terminal_flags[index - self.agent_history_length:index].any():
                    continue
                break
            self.indices[i] = index

    def get_minibatch(self):
        if self.count < self.agent_history_length:
            raise ValueError('Not enough memories to get a minibatch')

        self._get_valid_indices()

        for i, idx in enumerate(self.indices):
            self.states[i] = self.frames[idx-1]
            self.new_states[i] = self.frames[idx]

        return self.states, self.actions[self.indices], self.rewards[
            self.indices], self.new_states, self.terminal_flags[self.indices]