import ecole
import numpy as np
from collections import defaultdict


class NodeBipariteWith24VariableFeatures(ecole.observation.NodeBipartite):
    '''
    Adds (mostly global) features to variable node features.

    Adds 5 extra variable features to each variable on top of standard ecole
    NodeBipartite obs variable features (19), so each variable will have
    24 features in total.

    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def before_reset(self, model):
        super().before_reset(model)

        self.init_dual_bound = None
        self.init_primal_bound = None
        
        
    def extract(self, model, done):
        # get the NodeBipartite obs
        obs = super().extract(model, done)
        
        m = model.as_pyscipopt()
        
        if self.init_dual_bound is None:
            self.init_dual_bound = m.getDualbound()
            self.init_primal_bound = m.getPrimalbound()
            
        # dual/primal bound features
        # dual_bound_frac_change = abs(1-(min(self.init_dual_bound, m.getDualbound()) / max(self.init_dual_bound, m.getDualbound())))
        # primal_bound_frac_change = abs(1-(min(self.init_primal_bound, m.getPrimalbound()) / max(self.init_primal_bound, m.getPrimalbound())))
        dual_bound_frac_change = abs(self.init_dual_bound - m.getDualbound()) / self.init_dual_bound
        primal_bound_frac_change = abs(self.init_primal_bound - m.getPrimalbound()) / self.init_primal_bound

        primal_dual_gap = abs(m.getPrimalbound() - m.getDualbound())
        max_dual_bound_frac_change = primal_dual_gap / self.init_dual_bound
        max_primal_bound_frac_change = primal_dual_gap / self.init_primal_bound

        curr_primal_dual_bound_gap_frac = m.getGap()
        
        # add feats to each variable
        feats_to_add = np.array([[dual_bound_frac_change,
                                 primal_bound_frac_change,
                                 max_primal_bound_frac_change,
                                 max_dual_bound_frac_change,
                                 curr_primal_dual_bound_gap_frac,
                                 ] for _ in range(obs.variable_features.shape[0])])
        
        obs.variable_features = np.column_stack((obs.variable_features, feats_to_add))
                
        return obs


class ExploreThenStrongBranch:
    """
    This custom observation function class will randomly return either strong branching scores (expensive expert)
    or pseudocost scores (weak expert for exploration) when called at every node.
    """
    def __init__(self, expert_probability):
        self.expert_probability = expert_probability
        self.pseudocosts_function = ecole.observation.Pseudocosts()
        self.strong_branching_function = ecole.observation.StrongBranchingScores()

    def before_reset(self, model):
        """
        This function will be called at initialization of the environments (before dynamics are reset).
        """
        self.pseudocosts_function.before_reset(model)
        self.strong_branching_function.before_reset(model)

    def extract(self, model, done):
        """
        Should we return strong branching or pseudocost scores at time node?
        """
        probabilities = [1-self.expert_probability, self.expert_probability]
        expert_chosen = bool(np.random.choice(np.arange(2), p=probabilities))
        if expert_chosen:
            return (self.strong_branching_function.extract(model, done), True)
        else:
            return (self.pseudocosts_function.extract(model, done), False)


class PureStrongBranch:
    def __init__(self):
        self.strong_branching_function = ecole.observation.StrongBranchingScores()

    def before_reset(self, model):
        self.strong_branching_function.before_reset(model)

    def extract(self, model, done):
        return (self.strong_branching_function.extract(model, done))


class TripartiteNode:
    def __init__(self, observation_node, cut_features, cut_row_edge_features, cut_col_edge_features):
        self.variable_features = observation_node.variable_features
        self.row_features = observation_node.row_features
        self.edge_features = observation_node.edge_features
        self.cut_features = cut_features
        self.cut_row_edge_features = cut_row_edge_features
        self.cut_col_edge_features = cut_col_edge_features


class EdgeFeatures:
    def __init__(self, n, m):
        self.indices = [list(), list()]
        self.values = list()
        self.nnz = 0
        self.shape = (n, m)

    def add(self, ind_n, ind_m, value):
        self.indices[0].append(ind_n)
        self.indices[1].append(ind_m)
        self.values.append(value)
        self.nnz += 1

    def to_array(self):
        self.indices = np.array(self.indices)
        self.values = np.array(self.values)
        return self


def get_obj_norm(m):
    ob = m.getObjective()
    return np.sqrt(sum(x ** 2 for x in ob.terms.values()))


def get_extendet_rows(rows):
    ext_rows = list()
    for row in rows:
        if row.getLhs() > -1e20:
            ext_rows.append(row)
        if row.getRhs() < 1e20:
            ext_rows.append(row)
    return ext_rows


def generate_cut_features(col, obj_norm):
    bias = col.getPrimsol()
    objective_cosine_similarity = col.getObjCoeff() / obj_norm
    return [bias, objective_cosine_similarity]


def make_tripartite(env, node_observation, action_set):
    m = env.model.as_pyscipopt()

    n_cuts = len(action_set)
    n_rows, n_cols = node_observation.edge_features.shape
    cut_row_edge_features = EdgeFeatures(n_rows, n_cuts)
    cut_col_edge_features = EdgeFeatures(n_cuts, n_cols)
    cut_features = list()
    obj_norm = get_obj_norm(m)
    cut2col = dict()
    col2cut = dict()

    rows = m.getLPRowsData()
    ext_rows = get_extendet_rows(rows)
    cols = m.getLPColsData()
    row_inds, col_inds = node_observation.edge_features.indices

    for cut_ind, col_ind in enumerate(action_set):

        new_cut = generate_cut_features(cols[col_ind], obj_norm)
        cut_features.append(new_cut)

        cut_col_edge_features.add(cut_ind, col_ind, 1)

        non_orthogonal_inds = np.where(col_inds == col_ind)[0]
        non_orthogonal_rows = row_inds[non_orthogonal_inds]
        for row_ind in non_orthogonal_rows:
            scalar_product = 0

            row_ = ext_rows[row_ind]
            cols_ = row_.getCols()
            vals_ = row_.getVals()
            norm_ = row_.getNorm()
            for i, c in enumerate(cols_):
                if c.getLPPos() == col_ind:
                    scalar_product = abs(vals_[i] / norm_)

            cut_row_edge_features.add(cut_ind, row_ind, scalar_product)

    tripartite = TripartiteNode(node_observation,
                                np.array(cut_features),
                                cut_row_edge_features.to_array(),
                                cut_col_edge_features.to_array())
    return tripartite
