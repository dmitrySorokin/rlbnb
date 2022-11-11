import ecole
import glob

import numpy as np

from tasks import gen_co_name
from utils import get_most_recent_checkpoint_foldername
import hydra
from omegaconf import DictConfig
from agent import DQNAgent, StrongAgent
from env import EcoleBranching
import pandas as pd
from tqdm import trange


@hydra.main(config_path='configs', config_name='config.yaml')
def evaluate(cfg: DictConfig):
    files = glob.glob(f'../../../task_instances/{gen_co_name(cfg.instances.co_class, cfg.instances.co_class_kwargs)}/*.mps')
    instances = iter([ecole.scip.Model.from_file(f) for f in files])

    env = EcoleBranching(instances)
    env.seed(123)

    if cfg.agent.name == 'strong':
        agent = StrongAgent(env)
    elif cfg.agent.name == 'dqn':
        agent = DQNAgent(device='cpu')
        agent.eval()
        # checkpoint = get_most_recent_checkpoint_foldername('../../../outputs/02-37-47')
        print('eval checkpoint', cfg.agent.checkpoint)
        agent.load(f'../../../{cfg.agent.checkpoint}')
    elif cfg.agent.name == 'random':
        agent = DQNAgent(device='cpu')
        assert cfg.agent.epsilon == 1
    else:
        raise ValueError(f'Unknown agent name {cfg.agent.name}')

    df = pd.DataFrame(columns=['lp_iterations', 'num_nodes', 'solving_time'])

    for episode in trange(1000):
        obs, act_set, returns, done, info = env.eval_reset()
        while not done:
            action = agent.act(obs, act_set, epsilon=cfg.agent.epsilon)
            obs, act_set, returns, done, info = env.step(action)
        df = df.append(info, ignore_index=True)
        df.to_csv(f'../../../{cfg.output}')
        print(np.median(df['num_nodes']), np.std(df['num_nodes']))


if __name__ == '__main__':
    evaluate()
