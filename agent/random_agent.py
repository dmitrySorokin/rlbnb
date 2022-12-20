import numpy as np


class RandomAgent:
    def act(self, obs, action_set, deterministic, env):
        action = np.random.choice(action_set)
        return action, None
