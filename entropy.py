import ecole
import glob
import os
import gzip
import pickle

import numpy as np
import torch

from tasks import gen_co_name
from utils import get_most_recent_checkpoint_foldername, UnpackedTripartite, UnpackedBipartite
import hydra
from omegaconf import DictConfig
from agent import ImitationAgent, CrossEntropy
from env import EcoleBranching
import pandas as pd
from tqdm import trange
from obs import make_tripartite, PureStrongBranch, ExploreThenStrongBranch
from torch.nn.functional import softmax


@hydra.main(config_path='configs', config_name='config.yaml')
def evaluate(cfg: DictConfig):
    files = glob.glob(f'{cfg.experiment.test_instances}/*.mps')
    instances = iter([ecole.scip.Model.from_file(f) for f in files])
    scip_params = {'limits/time': 2 * 60}
    env = EcoleBranching(instances,
                         obs_function=(PureStrongBranch(),
                                       ExploreThenStrongBranch(expert_probability=0.3),
                                       ecole.observation.NodeBipartite()),
                         scip_params=scip_params)
    env.seed(123)

    bipartite_model = ImitationAgent(device=cfg.experiment.device,
                                     observation_format='bipartite',
                                     encode_possible_actions=False)
    bipartite_model.load(cfg.experiment.path_to_bipartite_model)
    bipartite_model.eval()
    masked_model = ImitationAgent(device=cfg.experiment.device,
                                  observation_format='bipartite',
                                  encode_possible_actions=True)
    masked_model.load(cfg.experiment.path_to_masked_bipartite_model)
    masked_model.eval()
    tripartite_model = ImitationAgent(device=cfg.experiment.device,
                                      observation_format='tripartite')
    tripartite_model.load(cfg.experiment.path_to_tripartite_model)
    tripartite_model.eval()
    loss = CrossEntropy().to(cfg.experiment.device)

    df = pd.DataFrame(columns=['bipartite_ce', 'bipartite_entropy', 'tripartite_ce', 'tripartite_entropy'])

    try:
        os.mkdir(f'{cfg.experiment.path_to_log}/logits')
    except FileExistsError:
        pass
    n = 0

    for episode in trange(1000):
        obs, act_set, returns, done, info = env.eval_reset()
        if not done:
            gt_scores, (pseudo_scores, _), obs = obs
            obs = make_tripartite(env, obs, act_set)
        while not done:
            target = torch.from_numpy(gt_scores[act_set]).argmax(dim=-1).to(cfg.experiment.device)

            obs_ = UnpackedBipartite(obs, act_set, cfg.experiment.device)
            bipartite_logits = bipartite_model.net(obs_)
            bipartite_p = softmax(bipartite_logits, dim=-1)

            masked_logits = masked_model.net(obs_)
            masked_p = softmax(masked_logits, dim=-1)

            obs_ = UnpackedTripartite(obs, cfg.experiment.device)
            tripartite_logits = tripartite_model.net(obs_)
            tripartite_p = softmax(tripartite_logits, dim=-1)

            filename = f'{cfg.experiment.path_to_log}/logits/{n}.pkl'
            data = [bipartite_logits, masked_logits, tripartite_logits]
            with gzip.open(filename, 'wb') as f:
                pickle.dump(data, f)
            n += 1

            bipartite_entropy = -torch.sum(bipartite_p * torch.log(bipartite_p + 1e-8))
            bipartite_loss = loss(bipartite_logits, target)
            masked_entropy = -torch.sum(masked_p * torch.log(masked_p + 1e-8))
            masked_loss = loss(masked_logits, target)
            tripartite_entropy = -torch.sum(tripartite_p * torch.log(tripartite_p + 1e-8))
            tripartite_loss = loss(tripartite_logits, target)
            info = {
                'bipartite_ce': bipartite_loss.cpu().item(),
                'bipartite_entropy': bipartite_entropy.cpu().item(),
                'masked_ce': masked_loss.cpu().item(),
                'masked_entropy': masked_entropy.cpu().item(),
                'tripartite_ce': tripartite_loss.cpu().item(),
                'tripartite_entropy': tripartite_entropy.cpu().item()
            }

            df = df.append(info, ignore_index=True)

            action = act_set[pseudo_scores[act_set].argmax()]
            obs, act_set, returns, done, info = env.step(action)
            if not done:
                gt_scores, (pseudo_scores, _), obs = obs
                obs = make_tripartite(env, obs, act_set)

        df.to_csv(f'{cfg.experiment.path_to_log}/tri_entropy_test.csv')


if __name__ == '__main__':
    evaluate()
