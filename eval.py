import ecole
import glob

import numpy as np

from tasks import gen_co_name
from utils import get_most_recent_checkpoint_foldername, UnpackedTripartite, UnpackedBipartite
import hydra
from omegaconf import DictConfig
from agent import DQNAgent, ImitationAgent, StrongAgent, RandomAgent
from env import EcoleBranching
import pandas as pd
from tqdm import trange
from obs import make_tripartite


@hydra.main(config_path='configs', config_name='config.yaml')
def evaluate(cfg: DictConfig):
    #files = glob.glob(f'../../../task_instances/{gen_co_name(cfg.instances.co_class, cfg.instances.co_class_kwargs)}/*.mps')
    files = glob.glob(f'{cfg.experiment.test_instances}/*.mps')
    instances = iter([ecole.scip.Model.from_file(f) for f in files])

    env = EcoleBranching(instances)
    env.seed(123)
    try:
        observation_format = cfg.learner.observation_format
    except:
        observation_format = 'bipartite'

    if cfg.agent.name == 'strong':
        agent = StrongAgent(env)
    elif cfg.agent.name == 'dqn':
        agent = DQNAgent(device=cfg.experiment.device)
        agent.eval()
        # checkpoint = get_most_recent_checkpoint_foldername('../../../outputs/02-37-47')
        print('eval checkpoint', cfg.agent.checkpoint)
        agent.load(f'../../../{cfg.agent.checkpoint}')
    elif cfg.agent.name == 'il':
        agent = ImitationAgent(device=cfg.experiment.device,
                               observation_format=observation_format)
        agent.eval()
        print('eval checkpoint', cfg.experiment.checkpoint)
        agent.load(f'{cfg.experiment.path_to_save}/checkpoint_{cfg.experiment.checkpoint}.pkl')
    elif cfg.agent.name == 'random':
        agent = RandomAgent(seed=cfg.experiment.seed)
        assert cfg.agent.epsilon == 1
    else:
        raise ValueError(f'Unknown agent name {cfg.agent.name}')

    df = pd.DataFrame(columns=['lp_iterations', 'num_nodes', 'solving_time'])

    for episode in trange(1000):
        obs, act_set, returns, done, info = env.eval_reset()
        if not done:
            obs = make_tripartite(env, obs, act_set)
        while not done:

            if observation_format == 'bipartite':
                obs = UnpackedBipartite(obs, act_set, cfg.experiment.device)
            elif observation_format == 'tripartite':
                obs = UnpackedTripartite(obs, cfg.experiment.device)

            action = agent.act(obs, act_set)
            obs, act_set, returns, done, info = env.step(action)
            if not done:
                obs = make_tripartite(env, obs, act_set)
        df = df.append(info, ignore_index=True)
        df.to_csv(f'{cfg.experiment.path_to_log}/test_result.csv')
        print(np.median(df['num_nodes']), np.std(df['num_nodes']))


if __name__ == '__main__':
    evaluate()
