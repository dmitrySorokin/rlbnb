from tasks import make_instances
from utils import seed_stochastic_modules_globally
import ecole
from pathlib import Path
import os
import random
import threading
from threading import Thread
import queue
from queue import Queue
from tqdm import trange
import hydra
from omegaconf import DictConfig, OmegaConf
from env import EcoleBranching
import numpy as np
hydra.HYDRA_FULL_ERROR = 1

n_parallel_process = 20


def strongbrancher(env: ecole.environment.Branching):
    """Get the strong branching policy for env"""
    sbf = ecole.observation.StrongBranchingScores()

    def _do_branch(obs: ..., act_set: ..., deterministic: bool = True) -> int:
        """Decide which variable to branch on using the SB heuristic"""
        scores = sbf.extract(env.model, False)[act_set]
        return act_set[np.argmax(scores)]

    return _do_branch


def run_sampler(cfg, path, sample_n_queue, seed=0):
    env = EcoleBranching(make_instances(cfg, seed=seed))
    env.seed(seed)
    pick = strongbrancher(env)

    while True:
        observation, action_set, _, done, info = env.reset()
        inst = info['instance']
        depth = -1
        while not done:
            action = pick(observation, action_set, deterministic=True)
            m = env.model.as_pyscipopt()
            depth = m.getDepth()
            observation, action_set, _, done, info = env.step(action)

        try:
            n = sample_n_queue.get(timeout=20)
        except queue.Empty:
            curr_thread = threading.current_thread()
            print(f'thread {curr_thread} finished')
            return

        inst.write_problem(f'{path}/instance_nodes_{int(info["num_nodes"])}_depth_{int(depth)}_{n}.mps')


def init_save_dir(path):
    _path = '../../../' + path
    Path(_path).mkdir(parents=True, exist_ok=True)
    return _path


@hydra.main(config_path='configs', config_name='config.yaml')
def run(cfg: DictConfig):
    # seeding
    if 'seed' not in cfg.experiment:
        cfg.experiment['seed'] = random.randint(0, 10000)
        seed_stochastic_modules_globally(cfg.experiment.seed)

    # print info
    print('\n\n\n')
    print(f'~'*80)
    print(f'Config:\n{OmegaConf.to_yaml(cfg)}')
    print(f'~'*80)

################
    path = cfg.experiment.path_to_load_imitation_data
    path = init_save_dir(path)
################

    print('Generating >={} samples in parallel on {} CPUs and saving to {}'.format(cfg.experiment.num_samples, n_parallel_process, os.path.abspath(path)))

    ecole.seed(cfg.experiment.seed)
    sample_n_queue = Queue(maxsize=64)

    threads = list()
    for i in range(n_parallel_process):
        process = Thread(target=run_sampler, args=(cfg, path, sample_n_queue, i))
        process.start()
        threads.append(process)

    for i in trange(cfg.experiment.num_samples):
        sample_n_queue.put(i)

    for process in threads:
        process.join()
    sample_n_queue.join()

if __name__ == '__main__':
    run()
