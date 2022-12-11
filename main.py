import hydra
import numpy as np
from omegaconf import DictConfig
from env import EcoleBranching
from tasks import make_instances
from agent import DQNAgent, StrongAgent, ReplayBuffer
from tqdm import tqdm
import os
from tensorboardX import SummaryWriter


def rollout(
    env: EcoleBranching,
    agent: ReplayBuffer,
    buffer: ReplayBuffer,
    epsilon: float,
    max_tree_size: int = 100,
) -> tuple[int, dict]:
    trajectory = []
    obs, act_set, returns, done, info = env.reset()
    while not done:
        action = agent.act(obs, act_set, epsilon)
        trajectory.append((obs, action))
        obs, act_set, returns, done, info = env.step(action)

    # the env outputs `returns` only on the last iteration
    traj_obs, traj_act = zip(*trajectory)
    tree_size = len(traj_obs)
    assert tree_size == len(returns)

    ids = np.random.choice(tree_size, min(tree_size, max_tree_size), replace=False)
    for j in ids:
        buffer.add_transition(traj_obs[j], traj_act[j], returns[j])

    return len(ids), info


@hydra.main(config_path="configs", config_name="retro.yaml")
def main(cfg: DictConfig):
    import pdb

    pdb.set_trace()

    writer = SummaryWriter(os.getcwd())

    env = EcoleBranching(
        make_instances(**cfg.instances), cfg.experiment.reward_function
    )
    env.seed(cfg.experiment.seed)

    agent = DQNAgent(device=cfg.experiment.device)
    agent.train()

    replay_buffer = ReplayBuffer(
        max_size=cfg.experiment.buffer_max_size,
        start_size=cfg.experiment.buffer_start_size,
    )

    epsilon_start = 1.0
    epsilon_min = 0.01
    epsilon = epsilon_start

    with tqdm(total=replay_buffer.start_size, desc="init") as pbar:
        while not replay_buffer.is_ready():
            num_obs, _ = rollout(env, agent, replay_buffer, epsilon)
            pbar.update(num_obs)

    pbar = tqdm(total=cfg.experiment.num_updates, desc="train")
    episode, update = 0, 0
    while update < pbar.total:
        num_obs, info = rollout(env, agent, replay_buffer, epsilon)
        writer.add_scalar("episode/num_nodes", info["num_nodes"], episode)
        writer.add_scalar("episode/lp_iterations", info["lp_iterations"], episode)
        writer.add_scalar("episode/solving_time", info["solving_time"], episode)
        print(episode, info["num_nodes"])
        episode += 1

        for i in range(min(num_obs, 100)):
            obs, act, ret = replay_buffer.sample()
            loss = agent.update(obs, act, ret)
            writer.add_scalar("update/loss", loss, update)
            writer.add_scalar("update/epsilon", epsilon, update)
            update += 1

            # epsilon = max(epsilon_min, epsilon * epsilon_decay)
            epsilon = max(epsilon_start * (1.0 - update / pbar.total), epsilon_min)
            print(f"loss = {loss:.2f}, epsilon = {epsilon:.2f}")

        agent.save(os.getcwd(), update)
        pbar.update(num_obs)

    pbar.close()


if __name__ == "__main__":
    main()
