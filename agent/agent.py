import torch
from torch import nn
import torch_geometric
import numpy as np
from torch_geometric.data import Batch  # noqa: F401

import ecole

from typing import NamedTuple
from ecole.observation import NodeBipartiteObs
from ecole.observation import coo_matrix as EdgeFeatures
from torch import Tensor, as_tensor


class EdgeObs(NamedTuple):
    inx: Tensor
    val: Tensor

    @classmethod
    def from_ecole(cls, obs: EdgeFeatures, device: torch.device = None) -> ...:
        return cls(
            inx=as_tensor(obs.indices.astype(int), dtype=torch.long, device=device),
            val=as_tensor(obs.values, dtype=torch.float32, device=device).view(-1, 1),
        )


class BipartiteObs(NamedTuple):
    vars: Tensor
    cons: Tensor
    ctov: EdgeObs

    @classmethod
    def from_ecole(cls, obs: NodeBipartiteObs, device: torch.device = None) -> ...:
        return cls(
            vars=as_tensor(obs.variable_features, dtype=torch.float32, device=device),
            cons=as_tensor(obs.row_features, dtype=torch.float32, device=device),
            ctov=EdgeObs.from_ecole(obs.edge_features, device),
        )


def init_linear(
    m: nn.Module, weight: str = None, nonlinearity: str = None, bias: str = None
) -> None:
    if not isinstance(m, nn.Linear):
        return

    if weight == "uniform":
        nn.init.uniform_(m.weight, a=0.0, b=1.0)

    elif weight == "normal":
        nn.init.normal_(m.weight, mean=0.0, std=0.01)

    elif weight == "xavier_uniform":
        gain = nn.init.calculate_gain(nonlinearity)
        nn.init.xavier_uniform_(m.weight, gain=gain)

    elif weight == "xavier_normal":
        gain = nn.init.calculate_gain(nonlinearity)
        nn.init.xavier_normal_(m.weight, gain=gain)

    elif weight == "kaiming_uniform":
        nn.init.kaiming_uniform_(m.weight, nonlinearity=nonlinearity)

    elif weight == "kaiming_normal":
        nn.init.kaiming_normal_(m.weight, nonlinearity=nonlinearity)
        # nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    elif weight is not None:
        raise NotImplementedError(f"Linear weight `{weight}`")

    if getattr(m, "bias", None) is None:
        return

    if bias == "zeros":
        nn.init.zeros_(m.bias)

    elif bias == "uniform":
        nn.init.uniform_(m.bias)

    elif bias == "normal":
        nn.init.normal_(m.bias)

    elif bias is not None:
        raise NotImplementedError(f"Linear bias `{bias}`")


def init_layernorm(m: nn.Module, weight: str = None, bias: str = None) -> None:
    if not isinstance(m, nn.LayerNorm):
        return

    if weight == "normal":
        nn.init.normal_(m.weight, mean=0.0, std=0.01)

    elif weight is not None:
        raise NotImplementedError(f"Layernorm weight `{weight}`")

    if bias == "zeros":
        nn.init.zeros_(m.bias)

    elif bias == "normal":
        nn.init.normal_(m.bias)

    elif bias is not None:
        raise NotImplementedError(f"Layernorm bias `{bias}`")


class BipartiteGraphConvolution(torch_geometric.nn.MessagePassing):
    """
    The bipartite graph convolution is already provided by pytorch geometric and we merely need
    to provide the exact form of the messages being passed.
    """

    def __init__(self, aggregator="mean", emb_size=64):
        super().__init__(aggregator)

        self.feature_module_left = nn.Sequential(nn.Linear(emb_size, emb_size))
        self.feature_module_right = nn.Sequential(
            nn.Linear(emb_size, emb_size, bias=False)
        )
        self.feature_module_final = nn.Sequential(
            nn.LayerNorm(emb_size),
            nn.LeakyReLU(),
            nn.Linear(emb_size, emb_size),
        )
        self.post_conv_module = nn.Sequential(nn.LayerNorm(emb_size))

        # output_layers
        self.output_module = nn.Sequential(
            nn.Linear(2 * emb_size, emb_size),
            # nn.LayerNorm(emb_size, emb_size), # added
            nn.LeakyReLU(),
            nn.Linear(emb_size, emb_size),
            # nn.LayerNorm(emb_size, emb_size), # added
        )

    def forward(self, left_features, edge_indices, edge_features, right_features):
        # def forward(self, left_features, edge_indices, right_features):
        """
        This method sends the messages, computed in the message method.
        """
        # output = self.propagate(edge_indices, size=(left_features.shape[0], right_features.shape[0]),
        # node_features=(left_features, right_features), edge_features=edge_features)
        # output = self.propagate(edge_indices, size=(left_features.shape[0], right_features.shape[0]),
        # node_features=(self.feature_module_left(left_features), self.feature_module_right(right_features)))
        output = self.propagate(
            edge_indices,
            size=(left_features.shape[0], right_features.shape[0]),
            node_features=(
                self.feature_module_left(left_features),
                self.feature_module_right(right_features),
            ),
            edge_features=None,
        )
        return self.output_module(
            torch.cat([self.post_conv_module(output), right_features], dim=-1)
        )

    def message(self, node_features_i, node_features_j, edge_features=None):
        # def message(self, node_features_i, node_features_j):
        # output = self.feature_module_final(self.feature_module_left(node_features_i)
        # # + self.feature_module_edge(edge_features)
        # + self.feature_module_right(node_features_j))
        # output = self.feature_module_final(node_features_i + node_features_j)
        if edge_features is not None:
            output = self.feature_module_final(
                node_features_i + node_features_j + edge_features
            )
        else:
            output = self.feature_module_final(node_features_i + node_features_j)
        return output


class BipartiteGCN(nn.Module):
    def __init__(
        self,
        device,
        emb_size: int = 64,
        cons_nfeats: int = 5,
        edge_nfeats: int = 1,
        var_nfeats: int = 43,
        num_heads: int = 1,
        head_depth: int = 1,
        linear_weight_init: str = None,
        linear_bias_init: str = None,
        layernorm_weight_init: str = None,
        layernorm_bias_init: str = None,
    ) -> None:
        super().__init__()

        self.device = device
        self.emb_size = emb_size
        self.cons_nfeats = cons_nfeats
        self.edge_nfeats = edge_nfeats
        self.var_nfeats = var_nfeats
        self.activation = None
        self.num_heads = num_heads
        self.head_depth = head_depth

        self.linear_weight_init = linear_weight_init
        self.linear_bias_init = linear_bias_init
        self.layernorm_weight_init = layernorm_weight_init
        self.layernorm_bias_init = layernorm_bias_init

        # CONSTRAINT EMBEDDING
        self.cons_embedding = nn.Sequential(
            nn.LayerNorm(cons_nfeats),
            nn.Linear(cons_nfeats, emb_size),
            # nn.LayerNorm(emb_size, emb_size), # added
            nn.LeakyReLU(),
            nn.Linear(emb_size, emb_size),
            # nn.LayerNorm(emb_size, emb_size), # added
            nn.LeakyReLU(),
        )

        # VARIABLE EMBEDDING
        self.var_embedding = nn.Sequential(
            nn.LayerNorm(var_nfeats),
            nn.Linear(var_nfeats, emb_size),
            # nn.LayerNorm(emb_size, emb_size), # added
            nn.LeakyReLU(),
            nn.Linear(emb_size, emb_size),
            # nn.LayerNorm(emb_size, emb_size), # added
            nn.LeakyReLU(),
        )

        self.conv_v_to_c = BipartiteGraphConvolution(
            emb_size=emb_size, aggregator="mean"
        )
        self.conv_c_to_v = BipartiteGraphConvolution(
            emb_size=emb_size, aggregator="mean"
        )

        heads = []
        for _ in range(self.num_heads):
            head = []
            for _ in range(self.head_depth):
                head.append(nn.Linear(emb_size, emb_size))
                head.append(nn.LeakyReLU())
            head.append(nn.Linear(emb_size, 1, bias=True))
            heads.append(nn.Sequential(*head))
        self.heads_module = nn.ModuleList(heads)

        self.init_model_parameters()
        self.to(device)

    def init_model_parameters(self, init_gnn_params=True, init_heads_params=True):
        def init_params(m):
            init_linear(
                m, self.linear_weight_init, self.activation, self.linear_bias_init
            )

            init_layernorm(m, self.layernorm_weight_init, self.layernorm_bias_init)

        if init_gnn_params:
            # init base GNN params
            self.apply(init_params)

        if init_heads_params:
            # init head output params
            for h in self.heads_module:
                h.apply(init_params)

    def forward(self, obs: NodeBipartiteObs) -> Tensor:
        obs = BipartiteObs.from_ecole(obs, self.device)
        assert obs.vars.shape[1] == self.var_nfeats

        # First step: linear embedding layers to a common dimension (64)
        cons = self.cons_embedding(obs.cons)
        vars = self.var_embedding(obs.vars)

        # Two half convolutions (message passing round)
        obs_vtoc_inx = torch.stack([obs.ctov.inx[1], obs.ctov.inx[0]], dim=0)
        cons = self.conv_v_to_c(vars, obs_vtoc_inx, obs.ctov.val, cons)
        vars = self.conv_c_to_v(cons, obs.ctov.inx, obs.ctov.val, vars)

        # get output for each head
        head_output = torch.stack(
            [head(vars).squeeze(-1) for head in self.heads_module], dim=0
        )

        return head_output.mean(dim=0)


class DQNAgent:
    def __init__(self, device, epsilon: float = None):
        self.net = BipartiteGCN(device=device, var_nfeats=24)
        self.opt = torch.optim.Adam(self.net.parameters())
        self.epsilon = epsilon

    def act(self, obs, action_set, epsilon: float = None):
        epsilon = float(self.epsilon if epsilon is None else epsilon)

        with torch.no_grad():
            preds = self.net(obs)[action_set.astype("int32")]

        # single observation
        if np.random.rand() < epsilon:
            action = np.random.choice(action_set)
        else:
            action_idx = torch.argmax(preds)
            action = action_set[action_idx.item()]
        return action

    def update(self, obs_batch, act_batch, ret_batch):
        self.opt.zero_grad()
        loss = 0
        norm_coef = 0
        # TODO use torch_geometric.data.Batch
        for obs, act, ret in zip(obs_batch, act_batch, ret_batch):
            pred = self.net(obs)[act]
            coef = np.abs(ret)
            loss += ((pred - ret) ** 2) * coef
            norm_coef += coef
        loss /= len(obs_batch) * norm_coef
        loss.backward()
        self.opt.step()
        return loss.detach().cpu().item()

    def save(self, path, epoch_id):
        torch.save(self.net.state_dict(), path + f"/checkpoint_{epoch_id}.pkl")

    def load(self, path):
        self.net.load_state_dict(torch.load(path, map_location=self.net.device))

    def train(self):
        self.net.train()

    def eval(self):
        self.net.eval()


class StrongAgent:
    def __init__(self, env, epsilon: float = None):
        self.sbf = ecole.observation.StrongBranchingScores()
        self.env = env

    def before_reset(self, model):
        self.sbf.before_reset(model)

    def act(self, obs, action_set, epsilon: float = None):
        scores = self.sbf.extract(self.env.model, False)[action_set]
        return action_set[np.argmax(scores)]

    def eval(self):
        pass
