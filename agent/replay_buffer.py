import numpy as np
import ecole


class ReplayBuffer:
    def __init__(self, max_size=50000, start_size=10, batch_size=32):
        self.max_size = max_size
        self.start_size = start_size
        self.size = 0
        self.insert_idx = 0
        self.batch_size = batch_size
        self.obs = np.zeros(max_size, dtype=ecole.core.observation.NodeBipartiteObs)
        self.act = np.zeros(max_size, int)
        self.ret = np.zeros(max_size, float)
        self.mask = np.zeros((max_size, 50), float)

    def add_transition(self, obs, act, ret, mask):
        self.insert_idx = self.insert_idx % self.max_size
        self.obs[self.insert_idx] = obs
        self.act[self.insert_idx] = act
        self.ret[self.insert_idx] = ret
        self.mask[self.insert_idx] = mask

        self.insert_idx += 1
        self.size = min(self.size + 1, self.max_size)

    def is_ready(self):
        return self.size >= self.start_size

    def sample(self):
        assert self.is_ready()

        ids = np.random.randint(0, self.size, self.batch_size)
        return self.obs[ids], self.act[ids], self.ret[ids], self.mask[ids]
