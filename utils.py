import torch
import numpy as np
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
