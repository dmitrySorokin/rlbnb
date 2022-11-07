import ecole

import hydra
from omegaconf import DictConfig, OmegaConf
from env import EcoleBranching
from utils import generate_tsp, generate_craballoc
from agent import Agent
from replay_buffer import ReplayBuffer
from tqdm import tqdm
import os
from tensorboardX import SummaryWriter


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
    return len(returns), info


@hydra.main(config_path='configs', config_name='config.yaml')
def main(cfg: DictConfig):
    writer = SummaryWriter(os.getcwd())

    env = EcoleBranching(make_instances(cfg))
    env.seed(cfg.experiment.seed)

    agent = Agent(device=cfg.experiment.device)
    agent.train()

    replay_buffer = ReplayBuffer(
        max_size=cfg.experiment.buffer_max_size,
        start_size=cfg.experiment.buffer_start_size
    )

    pbar = tqdm(total=replay_buffer.start_size, desc='init...')
    while not replay_buffer.is_ready():
        num_obs, _ = rollout(env, agent, replay_buffer)
        pbar.update(num_obs)
    pbar.close()

    pbar = tqdm(total=cfg.experiment.num_updates, desc='train...')
    update = 0
    episode = 0
    while update < pbar.total:
        num_obs, info = rollout(env, agent, replay_buffer)
        writer.add_scalar('episode/num_nodes', info['num_nodes'], episode)
        writer.add_scalar('episode/lp_iterations', info['lp_iterations'], episode)
        writer.add_scalar('episode/solving_time', info['solving_time'], episode)
        print(episode, info['num_nodes'])
        episode += 1
        for i in range(num_obs):
            obs, act, ret = replay_buffer.sample()
            loss = agent.update(obs, act, ret)
            writer.add_scalar('update/loss', loss, update)
            update += 1
            print(f'loss = {loss:.2f}')
        agent.save(os.getcwd(), update)
        pbar.update(num_obs)
    pbar.close()


if __name__ == '__main__':
    main()
