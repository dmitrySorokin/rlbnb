import numpy as np


class SumTree:
    """
    a binary tree data structure where the parent’s value is the sum of its children
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.size = 0
        self.insert_idx = 0

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, priority, data):
        idx = self.insert_idx + self.capacity - 1

        self.data[self.insert_idx] = data
        self.update(idx, priority)

        self.insert_idx = (self.insert_idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    # update priority
    def update(self, idx, priority):
        change = priority - self.tree[idx]

        self.tree[idx] = priority
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[dataIdx]

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)

        return self._retrieve(right, s - self.tree[left])
