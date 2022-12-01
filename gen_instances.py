from utils import seed_stochastic_modules_globally
from tasks import make_instances, gen_co_name
from tqdm import trange
from env import EcoleBranching
import hydra
from omegaconf import DictConfig, OmegaConf
import os


hydra.HYDRA_FULL_ERROR = 1
SEED = 42


@hydra.main(config_path='configs', config_name='config.yaml')
def run(cfg: DictConfig):
    seed_stochastic_modules_globally(SEED)

    print(f'~' * 80)
    print(f'Config:\n{OmegaConf.to_yaml(cfg)}')
    print(f'~' * 80)

    env = EcoleBranching(make_instances(cfg))
    env.seed(SEED)
    print(f'Initialised environment.')

    out_dir = '../../../task_instances/' + gen_co_name(cfg.instances.co_class, cfg.instances.co_class_kwargs)
    os.makedirs(out_dir, exist_ok=True)

    # data generation
    for i in trange(1000):
        obs, act, rew, done, info = env.reset()
        assert not done
        instance = info['instance']
        instance.write_problem(f'{out_dir}/instance_{i}.mps')


if __name__ == '__main__':
    run()
