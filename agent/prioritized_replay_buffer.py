import random
import numpy as np
from .sum_tree import SumTree


class PrioritizedReplayBuffer:
    e = 0.0001
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, max_size, start_size, batch_size=32):
        self.tree = SumTree(max_size)
        self.start_size = start_size
        self.batch_size = batch_size

    def add_transition(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def is_ready(self):
        return self.tree.size >= self.start_size

    def sample(self):
        batch = []
        idxs = []
        segment = self.tree.total() / self.batch_size
        priorities = np.zeros(self.batch_size)

        self.beta = min(1., self.beta + self.beta_increment_per_sampling)

        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities[i] = p
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.size * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a
