import ecole
import numpy as np


class StrongAgent:
    def __init__(self, env):
        self.strong_branching_function = ecole.observation.StrongBranchingScores()
        self.env = env

    def before_reset(self, model):
        self.strong_branching_function.before_reset(model)

    def act(self, obs, action_set, deterministic, env):
        scores = self.strong_branching_function.extract(self.env.model, False)[action_set]
        return action_set[np.argmax(scores)], None

    def eval(self):
        pass
