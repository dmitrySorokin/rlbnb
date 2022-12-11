import ecole
import glob
import numpy as np
from tasks import gen_co_name

# from utils import get_most_recent_checkpoint_foldername
import hydra
from omegaconf import DictConfig
from agent import DQNAgent, StrongAgent
from env import EcoleBranching
import pandas as pd
from tqdm import trange
import os


def make_agent(env: EcoleBranching, cfg: DictConfig) -> object:
    if cfg.name == "strong":
        return StrongAgent(env)

    if cfg.name == "dqn":
        checkpoint = cfg.get("_checkpoint", f"../../../{cfg.checkpoint}")
        print("eval checkpoint", checkpoint)

        agent = DQNAgent(device="cpu", epsilon=cfg.epsilon)
        agent.eval()
        agent.load(checkpoint)
        return agent

    if cfg.name == "random":
        assert cfg.epsilon == 1
        return DQNAgent(device="cpu", epsilon=cfg.epsilon)

    raise NotImplementedError(f"Agent `{cfg.name}`")


def evaluate(cfg: DictConfig):
    # process overrides
    co_name = gen_co_name(cfg.instances.co_class, cfg.instances.co_class_kwargs)
    basedir = cfg.get("_files", f"../../../task_instances/{co_name}")
    files = glob.glob(glob.os.path.join(basedir, "*.mps"))

    # get the env and the agent
    env = EcoleBranching([])
    env.seed(123)

    # checkpoint = get_most_recent_checkpoint_foldername('../../../outputs/02-37-47')
    agent = make_agent(env, cfg.agent)
    for instance in map(ecole.scip.Model.from_file, files):
        obs, act_set, _, done, info = env.base_reset(instance)
        while not done:
            act = agent.act(obs, act_set)
            obs, act_set, _, done, info = env.step(act)

        yield info


@hydra.main(config_path="configs", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    co_name = gen_co_name(cfg.instances.co_class, cfg.instances.co_class_kwargs)
    out_dir = cfg.get("_output", f"../../../results/{co_name}")
    os.makedirs(out_dir, exist_ok=True)

    output = f"{out_dir}/{cfg.agent.name}.csv"
    df = pd.DataFrame(columns=["lp_iterations", "num_nodes", "solving_time"])
    for _, info in zip(trange(100), evaluate(cfg)):
        df = df.append(info, ignore_index=True)
        df.to_csv(output)
        print(np.median(df["num_nodes"]), np.std(df["num_nodes"]))


if __name__ == "__main__":
    main()

"""bash
python eval.py --config-name="combinatorial_auction" \
   ++agent.name="strong" ++output="foo" \
   ++_files="/Users/ivannazarov/Github/repos_with_rl/copt/rlbnb/task_instances/combinatorial_auction_n_items_100_n_bids_500" \
   ++agent._checkpoint="/Users/ivannazarov/Github/repos_with_rl/copt/rlbnb/outputs/2022-12-09/01-46-41/checkpoint_999.pkl" \
   ++_output="/Users/ivannazarov/Github/repos_with_rl/copt/rlbnb/"

python main.py ++experiment.device="cpu" ++experiment.reward_function="lp-gains"
"""
