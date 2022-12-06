import torch
from torch import nn
import torch_geometric
import numpy as np
from torch_geometric.data import Batch
import ecole


class BipartiteGraphConvolution(torch_geometric.nn.MessagePassing):
    """
    The bipartite graph convolution is already provided by pytorch geometric and we merely need
    to provide the exact form of the messages being passed.
    """

    def __init__(self, aggregator='mean', emb_size=64):
        super().__init__(aggregator)

        self.feature_module_left = nn.Sequential(
            nn.Linear(emb_size, emb_size)
        )
        self.feature_module_right = nn.Sequential(
            nn.Linear(emb_size, emb_size, bias=False)
        )
        self.feature_module_final = nn.Sequential(
            nn.LayerNorm(emb_size),
            nn.LeakyReLU(),
            nn.Linear(emb_size, emb_size)
        )
        self.post_conv_module = nn.Sequential(
            nn.LayerNorm(emb_size)
        )

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
        output = self.propagate(edge_indices, size=(left_features.shape[0], right_features.shape[0]),
                                node_features=(
                                self.feature_module_left(left_features), self.feature_module_right(right_features)),
                                edge_features=None)
        return self.output_module(torch.cat([self.post_conv_module(output), right_features], dim=-1))

    def message(self, node_features_i, node_features_j, edge_features=None):
        # def message(self, node_features_i, node_features_j):
        # output = self.feature_module_final(self.feature_module_left(node_features_i)
        # # + self.feature_module_edge(edge_features)
        # + self.feature_module_right(node_features_j))
        # output = self.feature_module_final(node_features_i + node_features_j)
        if edge_features is not None:
            output = self.feature_module_final(node_features_i + node_features_j + edge_features)
        else:
            output = self.feature_module_final(node_features_i + node_features_j)
        return output


class BipartiteGCN(nn.Module):
    def __init__(self,
                 device,
                 emb_size=64,
                 cons_nfeats=5,
                 edge_nfeats=1,
                 var_nfeats=43):
        super().__init__()

        self.device = device
        self.emb_size = emb_size
        self.cons_nfeats = cons_nfeats
        self.edge_nfeats = edge_nfeats
        self.var_nfeats = var_nfeats

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

        self.conv_v_to_c = BipartiteGraphConvolution(emb_size=emb_size, aggregator='mean')
        self.conv_c_to_v = BipartiteGraphConvolution(emb_size=emb_size, aggregator='mean')

        self.heads = nn.ModuleList([nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.LeakyReLU(inplace=True),
            nn.Linear(emb_size, 1)
        ) for _ in range(50)])

        self.init_model_parameters()
        self.to(device)

    def init_model_parameters(self):

        def init_params(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.uniform_(m.bias)

        self.apply(init_params)

        for h in self.heads:
            h.apply(init_params)

    def forward(self, obs):
        constraint_features = torch.from_numpy(
            obs.row_features.astype(np.float32)
        ).to(self.device)

        edge_indices = torch.LongTensor(
            obs.edge_features.indices.astype(np.int16)
        ).to(self.device)

        edge_features = torch.from_numpy(
            obs.edge_features.values.astype(np.float32)
        ).view(-1, 1).to(self.device)

        variable_features = torch.from_numpy(
            obs.variable_features.astype(np.float32)
        ).to(self.device)

        # print(constraint_features.shape, edge_indices.shape, edge_features.shape, variable_features.shape)
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

        # First step: linear embedding layers to a common dimension (64)
        constraint_features = self.cons_embedding(constraint_features)
        assert variable_features.shape[1] == self.var_nfeats
        variable_features = self.var_embedding(variable_features)

        # Two half convolutions (message passing round)
        constraint_features = self.conv_v_to_c(
            variable_features, reversed_edge_indices, edge_features, constraint_features
        )
        variable_features = self.conv_c_to_v(
            constraint_features, edge_indices, edge_features, variable_features
        )

        # get output for each head
        masks = torch.FloatTensor(len(self.heads)).uniform_() > 0.5
        head_output = [
            head(variable_features).squeeze(-1)
            for mask, head in zip(masks, self.heads) if mask
        ]
        head_output = torch.stack(head_output, dim=0).mean(dim=0)

        return head_output


class DQNAgent:
    def __init__(self, device):
        self.net = BipartiteGCN(device=device, var_nfeats=24)
        self.opt = torch.optim.Adam(self.net.parameters())

    def act(self, obs, action_set, epsilon):
        with torch.no_grad():
            preds = self.net(obs)[action_set.astype('int32')]

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
        torch.save(self.net.state_dict(), path + f'/checkpoint_{epoch_id}.pkl')

    def load(self, path):
        self.net.load_state_dict(
            torch.load(path, map_location=self.net.device)
        )

    def train(self):
        self.net.train()

    def eval(self):
        self.net.eval()


class StrongAgent:
    def __init__(self, env):
        self.strong_branching_function = ecole.observation.StrongBranchingScores()
        self.env = env

    def before_reset(self, model):
        self.strong_branching_function.before_reset(model)

    def act(self, obs, action_set, epsilon):
        scores = self.strong_branching_function.extract(self.env.model, False)[action_set]
        return action_set[np.argmax(scores)]

    def eval(self):
        pass
