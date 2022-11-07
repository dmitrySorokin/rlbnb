import ecole

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from env import EcoleBranching
from utils import generate_tsp, generate_craballoc
from pprint import pprint
from agent import Agent
from replay_buffer import ReplayBuffer
from tqdm import tqdm, trange


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


def rollout(env, agent, replay_buffer):
    obs, act_set, returns, done, info = env.reset()
    while not done:
        action = agent.act(obs, act_set)
        replay_buffer.accumulate(obs, action)
        obs, act_set, returns, done, info = env.step(action)
    replay_buffer.add_returns(returns)
    return len(returns)


@hydra.main(config_path='configs', config_name='config.yaml')
def main(cfg: DictConfig):
    env = EcoleBranching(make_instances(cfg))
    agent = Agent(device='cpu')
    replay_buffer = ReplayBuffer()

    pbar = tqdm(total=replay_buffer.start_size)
    while not replay_buffer.is_ready():
        num_obs = rollout(env, agent, replay_buffer)
        pbar.update(num_obs)
    pbar.close()

    pbar = tqdm(total=10000)
    for epoch in range(pbar.total):
        obs, act, ret = replay_buffer.sample()
        loss = agent.update(obs, act, ret)

    print(replay_buffer.is_ready(), replay_buffer.size)


if __name__ == '__main__':
    main()
