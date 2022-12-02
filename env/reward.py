import numpy as np
from .normalised_lp_gain import NormalisedLPGain


def postorder(graph, root):
    tot = 0
    for child in graph.successors(root):
        tot += postorder(graph, child)

    graph.nodes[root]['subtree'] = tot + 1
    return graph.nodes[root]['subtree']


class RetroBranching:
    def __init__(self):
        self.normalised_lp_gain = NormalisedLPGain()
        self.debug_mode = False

    def before_reset(self, model):
        self.normalised_lp_gain.before_reset(model)

    def extract(self, model, done):
        # update normalised LP gain tracker
        _ = self.normalised_lp_gain.extract(model, done)

        if not done:
            return None

        # instance finished, retrospectively create subtree episode paths

        if self.normalised_lp_gain.tree.tree.graph['root_node'] is None:
            # instance was pre-solved
            return {0: 0}

        # remove nodes which were never visited by the brancher and therefore do not have a score or next state
        nodes = [node for node in self.normalised_lp_gain.tree.tree.nodes]
        for node in nodes:
            if 'score' not in self.normalised_lp_gain.tree.tree.nodes[node]:
                self.normalised_lp_gain.tree.tree.remove_node(node)
                if node in self.normalised_lp_gain.tree.tree.graph['visited_node_ids']:
                    # hack: SCIP sometimes returns large int rather than None node_id when episode finished
                    # since never visited this node (since no score assigned), do not count this node as having been visited when calculating paths below
                    if self.debug_mode:
                        print(f'Removing node {node} since was never visited by brancher.')
                    self.normalised_lp_gain.tree.tree.graph['visited_node_ids'].remove(node)

        if len(self.normalised_lp_gain.tree.tree) == 0:
            return {0: 0}

        postorder(
            self.normalised_lp_gain.tree.tree,
            list(self.normalised_lp_gain.tree.tree.graph['root_node'].keys())[0]
        )

        rewards = []
        for node in self.normalised_lp_gain.tree.tree.graph['visited_node_ids']:
            rewards.append(-np.log(self.normalised_lp_gain.tree.tree.nodes[node]['subtree']))

        if self.debug_mode:
            print('\nB&B tree:')
            print(f'All nodes saved: {self.normalised_lp_gain.tree.tree.nodes()}')
            print(f'Visited nodes: {self.normalised_lp_gain.tree.tree.graph["visited_node_ids"]}')
            print(rewards)
            self.normalised_lp_gain.tree.render()

        return rewards
