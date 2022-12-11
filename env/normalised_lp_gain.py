from .utils import SearchTree
import copy


class NormalisedLPGain:
    def __init__(self):
        self.prev_node = None
        self.prev_node_id = None
        self.prev_primal_bound = None
        self.init_primal_bound = None
        self.tree = None

    def before_reset(self, model):
        self.prev_node = None
        self.prev_node_id = None
        self.prev_primal_bound = None
        self.init_primal_bound = None
        self.tree = None

    def extract(self, model, done):
        m = model.as_pyscipopt()

        if self.prev_node_id is None:
            # not yet started, update prev node for next step
            self.prev_node = m.getCurrentNode()
            self.tree = SearchTree(model)
            if self.prev_node is not None:
                self.prev_node_id = copy.deepcopy(self.prev_node.getNumber())
                self.prev_primal_bound = m.getPrimalbound()
                self.init_primal_bound = m.getPrimalbound()
            return 0

        # update search tree with current model state
        self.tree.update_tree(model)

        # collect node stats from children introduced from previous branching decision
        prev_node_lb = self.tree.tree.nodes[self.prev_node_id]["lower_bound"]
        prev_node_child_ids = [
            child for child in self.tree.tree.successors(self.prev_node_id)
        ]
        prev_node_child_lbs = [
            self.tree.tree.nodes[child]["lower_bound"] for child in prev_node_child_ids
        ]

        # calc reward for previous branching decision
        if len(prev_node_child_lbs) > 0:
            # use child lp gains to retrospectively calculate a score for the previous branching decision
            closed_by_agent = False
            score = -1
        else:
            # previous branching decision led to all child nodes being pruned, infeasible, or outside bounds -> don't punish brancher
            closed_by_agent = True
            score = 0

        # update tree with effect(s) of branching decision
        self.tree.tree.nodes[self.prev_node_id]["score"] = score
        self.tree.tree.nodes[self.prev_node_id]["closed_by_agent"] = closed_by_agent

        if m.getCurrentNode() is not None:
            # update stats for next step
            self.prev_node = m.getCurrentNode()
            self.prev_node_id = copy.deepcopy(self.prev_node.getNumber())
            self.prev_primal_bound = m.getPrimalbound()
        else:
            # instance completed, no current focus node
            pass

        return score
