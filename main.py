import ecole

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from env import EcoleBranching
from tasks import generate_tsp, generate_craballoc
from agent import DQNAgent, StrongAgent
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


def rollout(env, agent, replay_buffer, epsilon, max_tree_size=100):
    obs, act_set, returns, done, info = env.reset()
    traj_obs, traj_act = [], []

    while not done:
        action = agent.act(obs, act_set, epsilon)
        traj_obs.append(obs)
        traj_act.append(action)
        obs, act_set, returns, done, info = env.step(action)

    assert len(traj_obs) == len(returns)
    tree_size = len(traj_obs)
    ids = np.random.choice(range(tree_size), min(tree_size, max_tree_size), replace=False)
    traj_obs = np.asarray(traj_obs)[ids]
    traj_act = np.asarray(traj_act)[ids]
    traj_ret = np.asarray(returns)[ids]
    for obs, act, ret in zip(traj_obs, traj_act, traj_ret):
        replay_buffer.add_transition(obs, act, ret)

    return len(returns), info


@hydra.main(config_path='configs', config_name='retro.yaml')
def main(cfg: DictConfig):
    writer = SummaryWriter(os.getcwd())

    env = EcoleBranching(make_instances(cfg))
    env.seed(cfg.experiment.seed)

    agent = DQNAgent(device=cfg.experiment.device)
    agent.train()

    replay_buffer = ReplayBuffer(
        max_size=cfg.experiment.buffer_max_size,
        start_size=cfg.experiment.buffer_start_size
    )

    epsilon = 1
    epsilon_min = 0.01
    epsilon_decay = cfg.experiment.epsilon_decay

    pbar = tqdm(total=replay_buffer.start_size, desc='init')
    while not replay_buffer.is_ready():
        num_obs, _ = rollout(env, agent, replay_buffer, epsilon)
        pbar.update(num_obs)
    pbar.close()

    pbar = tqdm(total=cfg.experiment.num_updates, desc='train')
    update = 0
    episode = 0
    while update < pbar.total:
        num_obs, info = rollout(env, agent, replay_buffer, epsilon)
        writer.add_scalar('episode/num_nodes', info['num_nodes'], episode)
        writer.add_scalar('episode/lp_iterations', info['lp_iterations'], episode)
        writer.add_scalar('episode/solving_time', info['solving_time'], episode)
        print(episode, info['num_nodes'])
        episode += 1
        for i in range(min(num_obs, 100)):
            obs, act, ret = replay_buffer.sample()
            loss = agent.update(obs, act, ret)
            writer.add_scalar('update/loss', loss, update)
            writer.add_scalar('update/epsilon', epsilon, update)
            update += 1
            epsilon = max(epsilon_min, epsilon * epsilon_decay)
            print(f'loss = {loss:.2f}, epsilon = {epsilon:.2f}')
        agent.save(os.getcwd(), update)
        pbar.update(num_obs)
    pbar.close()


if __name__ == '__main__':
    main()
