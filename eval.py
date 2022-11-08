import ecole
import glob

import numpy as np

from tasks import gen_co_name
from utils import get_most_recent_checkpoint_foldername
import hydra
from omegaconf import DictConfig
from agent import Agent
from env import EcoleBranching


@hydra.main(config_path='configs', config_name='retro.yaml')
def evaluate(cfg: DictConfig):
    files = glob.glob(f'../../../task_instances/{gen_co_name(cfg.instances.co_class, cfg.instances.co_class_kwargs)}/*.mps')
    instances = iter([ecole.scip.Model.from_file(f) for f in files])

    agent = Agent(device='cpu')
    agent.eval()

    checkpoint = get_most_recent_checkpoint_foldername('../../../outputs/02-37-47')
    print('eval checkpoint', checkpoint)
    agent.load(f'../../../outputs/02-37-47/{checkpoint}')

    env = EcoleBranching(instances)
    env.seed(123)

    nodes = []
    for episode in range(100):
        obs, act_set, returns, done, info = env.eval_reset()
        while not done:
            action = agent.act(obs, act_set, epsilon=0)
            obs, act_set, returns, done, info = env.step(action)
        nodes.append(info['num_nodes'])
        print(episode, info)
        print(np.median(nodes), np.std(nodes))
    print(np.median(nodes), np.std(nodes))


if __name__ == '__main__':
    evaluate()
