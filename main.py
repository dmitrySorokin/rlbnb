import ecole
import torch

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from env import EcoleBranching
from utils import generate_tsp, generate_craballoc
import numpy as np
from pprint import pprint
from agent import Agent


def make_instances(cfg: DictConfig):
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

    return instances


@hydra.main(config_path='configs', config_name='config.yaml')
def main(cfg: DictConfig):
    env = EcoleBranching(make_instances(cfg))
    obs, act_set, reward, done, info = env.reset()
    agent = Agent(device='cpu')

    while not done:
        action = agent.act(obs, act_set)
        obs, act_set, reward, done, info = env.step(action)

    pprint(reward)
    print(info)


if __name__ == '__main__':
    main()
