import torch
import torch.nn.functional as F
import torch_geometric
import numpy as np
import gzip
import pickle
import ecole
import os
import random
import pyscipopt


def seed_stochastic_modules_globally(default_seed=0,
                                     numpy_seed=None,
                                     random_seed=None,
                                     torch_seed=None,
                                     ecole_seed=None):
    '''Seeds any stochastic modules so get reproducible results.'''
    if numpy_seed is None:
        numpy_seed = default_seed
    if random_seed is None:
        random_seed = default_seed
    if torch_seed is None:
        torch_seed = default_seed
    if ecole_seed is None:
        ecole_seed = default_seed

    np.random.seed(numpy_seed)

    random.seed(random_seed)

    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed(torch_seed)
    torch.cuda.manual_seed_all(torch_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    ecole.seed(ecole_seed)


def turn_off_scip_heuristics(ecole_instance):
    # write ecole instance to mps
    ecole_instance.write_problem('tmp_instance.mps')

    # read mps into pyscip model
    pyscipopt_instance = pyscipopt.Model()
    pyscipopt_instance.readProblem('tmp_instance.mps')

    # turn off heuristics
    pyscipopt_instance.setPresolve(pyscipopt.SCIP_PARAMSETTING.OFF)
    pyscipopt_instance.setHeuristics(pyscipopt.SCIP_PARAMSETTING.OFF)
    pyscipopt_instance.disablePropagation()
    pyscipopt_instance.setSeparating(pyscipopt.SCIP_PARAMSETTING.OFF)

    return ecole.scip.Model.from_pyscipopt(pyscipopt_instance)

def get_most_recent_checkpoint_foldername(path, idx=-1):
    '''
    Given a path to a folders named <name>_<number>, will sort checkpoints (i.e. most recently saved checkpoint
    last) and return name of idx provided (default idx=-1 to return
    most recent checkpoint folder).
    '''
    foldernames = [name.split('_') for name in os.listdir(path)]
    idx_to_num = {idx: int(num[:-4]) for idx, num in zip(range(len(foldernames)), [name[1] for name in foldernames if name[0] == 'checkpoint'])}
    sorted_indices = np.argsort(list(idx_to_num.values()))
    _idx = sorted_indices[idx]
    foldername = [name for name in os.listdir(path) if name.split('_')[0] == 'checkpoint'][_idx]
    return foldername

def pad_tensor(input_, pad_sizes, pad_value=-1e8):
    """
    This utility function splits a tensor and pads each split to make them all the same size, then stacks them.
    """
    max_pad_size = pad_sizes.max()
    output = input_.split(pad_sizes.cpu().numpy().tolist())
    output = torch.stack([F.pad(slice_, (0, max_pad_size - slice_.size(0)), 'constant', pad_value)
                          for slice_ in output], dim=0)
    return output


########################### PYTORCH DATA LOADERS #############################
class BipartiteNodeData(torch_geometric.data.Data):
    """
    This class encode a node bipartite graph observation as returned by the `ecole.observation.NodeBipartite` 
    observation function in a format understood by the pytorch geometric data handlers.
    """
    def __init__(self, 
                 observation=None,
                 candidates=None,
                 candidate_choice=None,
                 candidate_scores=None,
                 score=None):
        super().__init__()

        if observation is not None:
            self.row_features = torch.FloatTensor(observation.row_features)
            self.variable_features = torch.FloatTensor(observation.variable_features)
            self.edge_index = torch.LongTensor(observation.edge_features.indices.astype(np.int64))
            self.edge_attr = torch.FloatTensor(observation.edge_features.values).unsqueeze(1)
        if candidates is not None:
            self.candidates = torch.LongTensor(candidates)
            self.num_candidates = len(candidates)
        if candidate_choice is not None:
            self.candidate_choices = torch.LongTensor(candidate_choice)
        if candidate_scores is not None:
            self.candidate_scores = torch.FloatTensor(candidate_scores)
        if score is not None:
            self.score = torch.FloatTensor(score)

    def __inc__(self, key, value, *args, **kwargs):
        """
        We overload the pytorch geometric method that tells how to increment indices when concatenating graphs 
        for those entries (edge index, candidates) for which this is not obvious.
        """
        if key == 'edge_index':
            return torch.tensor([[self.row_features.size(0)], [self.variable_features.size(0)]])
        elif key == 'candidates':
            return self.variable_features.size(0)
        else:
            return super().__inc__(key, value)


class UnpackedBipartite:
    def __init__(self, observation, candidates, device):
        self.row_features = torch.FloatTensor(observation.row_features).to(device)
        self.variable_features = torch.FloatTensor(observation.variable_features).to(device)
        self.edge_index = torch.LongTensor(observation.edge_features.indices.astype(np.int64)).to(device)
        self.edge_attr = torch.FloatTensor(observation.edge_features.values).unsqueeze(1).to(device)
        self.candidates = torch.LongTensor(candidates.astype('int64'))


class UnpackedTripartite:
    def __init__(self, observation, device):
        self.row_features = torch.FloatTensor(observation.row_features).to(device)
        self.variable_features = torch.FloatTensor(observation.variable_features).to(device)
        self.cut_features = torch.FloatTensor(observation.cut_features).to(device)
        self.edge_index = torch.LongTensor(observation.edge_features.indices.astype(np.int64)).to(device)
        self.edge_features = torch.FloatTensor(observation.edge_features.values).unsqueeze(1).to(device)
        self.cut_col_edge_index = torch.LongTensor(observation.cut_col_edge_features.indices.astype(np.int64)).to(device)
        self.cut_col_edge_features = torch.FloatTensor(observation.cut_col_edge_features.values).unsqueeze(1).to(device)
        self.cut_row_edge_index = torch.LongTensor(observation.cut_row_edge_features.indices.astype(np.int64)).to(device)
        self.cut_row_edge_features = torch.FloatTensor(observation.cut_row_edge_features.values).unsqueeze(1).to(device)


class TripartiteNodeData(torch_geometric.data.Data):
    def __init__(self,
                 observation=None,
                 candidates=None,
                 candidate_choice=None,
                 candidate_scores=None,
                 score=None):
        super().__init__()

        if observation is not None:
            self.row_features = torch.FloatTensor(observation.row_features)
            self.variable_features = torch.FloatTensor(observation.variable_features)
            self.cut_features = torch.FloatTensor(observation.cut_features)
            self.edge_index = torch.LongTensor(observation.edge_features.indices.astype(np.int64))
            self.edge_features = torch.FloatTensor(observation.edge_features.values).unsqueeze(1)
            self.cut_col_edge_index = torch.LongTensor(observation.cut_col_edge_features.indices.astype(np.int64))
            self.cut_col_edge_features = torch.FloatTensor(observation.cut_col_edge_features.values).unsqueeze(1)
            self.cut_row_edge_index = torch.LongTensor(observation.cut_row_edge_features.indices.astype(np.int64))
            self.cut_row_edge_features = torch.FloatTensor(observation.cut_row_edge_features.values).unsqueeze(1)
        if candidates is not None:
            self.candidates = torch.LongTensor(candidates)
            self.num_candidates = len(candidates)
        if candidate_choice is not None:
            self.candidate_choices = torch.LongTensor(candidate_choice)
        if candidate_scores is not None:
            self.candidate_scores = torch.FloatTensor(candidate_scores)
        if score is not None:
            self.score = torch.FloatTensor(score)

    def __inc__(self, key, value, *args, **kwargs):
        """
        We overload the pytorch geometric method that tells how to increment indices when concatenating graphs
        for those entries (edge index, candidates) for which this is not obvious.
        """
        #print(key)
        if key == 'edge_index':
        #    print(self.row_features.size(0), self.variable_features.size(0))
            return torch.tensor([[self.row_features.size(0)], [self.variable_features.size(0)]])
        if key == 'cut_col_edge_index':
        #    #print(self.cut_features.size(0), self.variable_features.size(0))
            return torch.tensor([[self.cut_features.size(0)], [self.variable_features.size(0)]])
        if key == 'cut_row_edge_index':
            #print(self.cut_features.size(0), self.row_features.size(0))
            return torch.tensor([[self.cut_features.size(0)], [self.row_features.size(0)]])
        if key == 'candidates':
            #print(self.variable_features.size(0))
            return self.variable_features.size(0)
        else:
            return super().__inc__(key, value)


class GraphDataset(torch_geometric.data.Dataset):
    """
    This class encodes a collection of graphs, as well as a method to load such graphs from the disk.
    It can be used in turn by the data loaders provided by pytorch geometric.
    """
    def __init__(self, sample_files, observation_format='bipartite'):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.sample_files = sample_files
        assert observation_format in ['bipartite', 'tripartite'], 'Not implemented'
        self.observation_node = BipartiteNodeData if observation_format == 'bipartite' else TripartiteNodeData

        self.get_num_nodes = (lambda obs: obs.row_features.shape[0]+obs.variable_features.shape[0])\
                             if observation_format == 'bipartite' else \
                             (lambda obs: obs.row_features.shape[0]+obs.variable_features.shape[0]+obs.cut_features.shape[0])

    def len(self):
        return len(self.sample_files)

    def get(self, index):
        """
        This method loads a node bipartite graph observation as saved on the disk during data collection.
        """
        with gzip.open(self.sample_files[index], 'rb') as f:
            sample = pickle.load(f)

        sample_observation, sample_action, sample_action_set, sample_scores = sample
        
        # We note on which variables we were allowed to branch, the scores as well as the choice 
        # taken by strong branching (relative to the candidates)
        candidates = torch.LongTensor(np.array(sample_action_set, dtype=np.int32))
        try:
            candidate_scores = torch.FloatTensor([sample_scores[j] for j in candidates])
            score = []
        except (TypeError, IndexError):
            # only given one score and not in a list so not iterable
            score = torch.FloatTensor([sample_scores])
            candidate_scores = []
        candidate_choice = torch.where(candidates == sample_action)[0][0]

        graph = self.observation_node(sample_observation, candidates, candidate_choice, candidate_scores, score)
        
        # We must tell pytorch geometric how many nodes there are, for indexing purposes
        graph.num_nodes = self.get_num_nodes(sample_observation)
        
        return graph
