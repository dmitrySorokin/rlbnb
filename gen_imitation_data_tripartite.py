from tasks import make_instances
from utils import seed_stochastic_modules_globally
from env import make_tripartite, ExploreThenStrongBranch, PureStrongBranch
import ecole
import gzip
import pickle
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
hydra.HYDRA_FULL_ERROR = 1

n_parallel_process = 16
scip_params = {
    'separating/maxrounds': 0,  # separate (cut) only at root node
    'presolving/maxrestarts': 0,  # disable solver restarts
    'limits/time': 60 * 60,  # solver time limit
    'timing/clocktype': 1,  # 1: CPU user seconds, 2: wall clock time
    # 'limits/gap': 3e-4,  # 0.03% relative primal-dual gap (default: 0.0)
    # 'limits/nodes': -1,
}

def run_sampler(cfg, path, sample_n_queue, seed=0):
    instance_gen = make_instances(cfg, seed=seed)
    branching = cfg.experiment.branching
    max_steps = cfg.experiment.max_steps

    if branching == 'explore_then_strong_branch':
        env = ecole.environment.Branching(observation_function=(ExploreThenStrongBranch(expert_probability=0.3),
                                                                ecole.observation.NodeBipartite()),
                                          scip_params=scip_params)
    elif branching == 'pure_strong_branch':
        env = ecole.environment.Branching(observation_function=(PureStrongBranch(),
                                                                ecole.observation.NodeBipartite()),
                                          scip_params=scip_params)
    else:
        raise Exception('Unrecognised branching {}'.format(branching))

    n = sample_n_queue.get(timeout=5)

    while True:
        instance = next(instance_gen)
        observation, action_set, _, done, _ = env.reset(instance)
        t = 0
        while not done:
            if branching == 'explore_then_strong_branch':
                # only save samples if they are coming from the expert (strong branching)
                (scores, save_samples), node_observation = observation
            elif branching == 'pure_strong_branch':
                # always save samples since always using strong branching
                save_samples = True
                scores, node_observation = observation
            else:
                raise Exception('Unrecognised branching {}'.format(branching))

            action = action_set[scores[action_set].argmax()]

            if save_samples:
                node_tripartite = make_tripartite(env, node_observation, action_set)
                data = [node_tripartite, action, action_set, scores]
                filename = f'{path}sample_{n}.pkl'
                with gzip.open(filename, 'wb') as f:
                    pickle.dump(data, f)
                sample_n_queue.task_done()

                try:
                    n = sample_n_queue.get(timeout=5)
                except queue.Empty:
                    curr_thread = threading.current_thread()
                    print(f'thread {curr_thread} finished')
                    return

            observation, action_set, _, done, _ = env.step(action)
            t += 1
            if max_steps is not None:
                if t >= max_steps:
                    # stop episode
                    break

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
        process = Thread(target=run_sampler, args=(path, sample_n_queue, cfg, i))
        process.start()
        threads.append(process)

    for i in trange(cfg.experiment.num_samples):
        sample_n_queue.put(i)

    for process in threads:
        process.join()
    sample_n_queue.join()

if __name__ == '__main__':
    run()