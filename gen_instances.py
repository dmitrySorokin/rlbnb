from utils import seed_stochastic_modules_globally
from obs import PureStrongBranch
from tasks import generate_tsp, generate_craballoc
import ecole

import random
from tqdm import trange

import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

hydra.HYDRA_FULL_ERROR = 1
scip_params = {'separating/maxrounds': 0,  # separate (cut) only at root node
               'presolving/maxrestarts': 0,  # disable solver restarts
               'limits/time': 20*60,  # solver time limit
               'timing/clocktype': 2,
               'limits/gap': 3e-4,
               'limits/nodes': -1}


def init_save_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


@hydra.main(config_path='configs', config_name='config.yaml')
def run(cfg: DictConfig):
    # seeding
    if 'seed'in cfg.experiment:
        seed = cfg.experiment['seed']
    else:
        seed = random.randint(0, 10000)
    seed_stochastic_modules_globally(seed)

    # print info
    print('\n\n\n')
    print(f'~' * 80)
    print(f'Config:\n{OmegaConf.to_yaml(cfg)}')
    print(f'~' * 80)

    # initialise instance generator
    if 'path_to_instances' in cfg.instances:
        instances = ecole.instance.FileGenerator(cfg.instances.path_to_instances,
                                                 sampling_mode=cfg.instances.sampling_mode)
    else:
        if cfg.instances.co_class == 'set_covering':
            instances = ecole.instance.SetCoverGenerator(**cfg.instances.co_class_kwargs)
        elif cfg.instances.co_class == 'combinatorial_auction':
            instances = ecole.instance.CombinatorialAuctionGenerator(**cfg.instances.co_class_kwargs)
        elif cfg.instances.co_class == 'capacitated_facility_location':
            instances = ecole.instance.CapacitatedFacilityLocationGenerator(**cfg.instances.co_class_kwargs)
        elif cfg.instances.co_class == 'maximum_independent_set':
            instances = ecole.instance.IndependentSetGenerator(**cfg.instances.co_class_kwargs)
        elif cfg.instances.co_class == 'crabs':
            instances = generate_craballoc(**cfg.instances.co_class_kwargs)
        elif cfg.instances.co_class == 'tsp':
            instances = generate_tsp(**cfg.instances.co_class_kwargs)
        else:
            raise Exception(f'Unrecognised co_class {cfg.instances.co_class}')
    print(f'Initialised instance generator.')

    # initialise branch-and-bound environment
    env = ecole.environment.Branching(observation_function=(PureStrongBranch(),
                                                            ecole.observation.NodeBipartite()),
                                          scip_params=scip_params)
    print(f'Initialised environment.')
    
    init_save_dir(cfg.experiment.path_to_load_instances)
    # data generation
    for i in trange(1000):
        done = True
        while done:
            instance = next(instances)
            obs, act, rew, done, info = env.reset(instance.copy_orig())
        instance.write_problem(
            f'{cfg.experiment.path_to_load_instances}/'
            f'/{i}.mps'
        )


if __name__ == '__main__':
    run()
