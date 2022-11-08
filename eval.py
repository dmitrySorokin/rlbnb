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


@hydra.main(config_path='configs', config_name='retro.yaml')
def evaluate(cfg: DictConfig):
    files = glob.glob(f'../../../task_instances/{gen_co_name(cfg.instances.co_class, cfg.instances.co_class_kwargs)}/*.mps')
    instances = iter([ecole.scip.Model.from_file(f) for f in files])

    agent = DQNAgent(device='cpu')
    agent.eval()

    checkpoint = get_most_recent_checkpoint_foldername('../../../outputs/02-37-47')
    print('eval checkpoint', checkpoint)
    agent.load(f'../../../outputs/02-37-47/{checkpoint}')

    env = EcoleBranching(instances)
    env.seed(123)

    df = pd.DataFrame(columns=['lp_iterations', 'num_nodes', 'solving_time'])

    for episode in trange(100):
        obs, act_set, returns, done, info = env.eval_reset()
        while not done:
            action = agent.act(obs, act_set, epsilon=0)
            obs, act_set, returns, done, info = env.step(action)
        df = df.append(info, ignore_index=True)
        df.to_csv('../../../results/retro.csv')
        print(np.median(df['num_nodes']), np.std(df['num_nodes']))


if __name__ == '__main__':
    evaluate()
