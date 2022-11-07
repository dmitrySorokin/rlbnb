import torch
import torch_geometric
import numpy as np
from torch_geometric.data import Batch


class BipartiteGraphConvolution(torch_geometric.nn.MessagePassing):
    """
    The bipartite graph convolution is already provided by pytorch geometric and we merely need
    to provide the exact form of the messages being passed.
    """

    def __init__(self, aggregator='mean', emb_size=64):
        super().__init__(aggregator)

        self.feature_module_left = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size)
        )
        self.feature_module_right = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size, bias=False)
        )
        self.feature_module_final = torch.nn.Sequential(
            torch.nn.LayerNorm(emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, emb_size)
        )
        self.post_conv_module = torch.nn.Sequential(
            torch.nn.LayerNorm(emb_size)
        )

        # output_layers
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(2 * emb_size, emb_size),
            # torch.nn.LayerNorm(emb_size, emb_size), # added
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, emb_size),
            # torch.nn.LayerNorm(emb_size, emb_size), # added
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


class BipartiteGCN(torch.nn.Module):
    def __init__(self,
                 device,
                 emb_size=64,
                 cons_nfeats=5,
                 edge_nfeats=1,
                 var_nfeats=43,
                 num_heads=1,
                 head_depth=1,
                 linear_weight_init=None,
                 linear_bias_init=None,
                 layernorm_weight_init=None,
                 layernorm_bias_init=None):
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
        self.cons_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(cons_nfeats),
            torch.nn.Linear(cons_nfeats, emb_size),
            # torch.nn.LayerNorm(emb_size, emb_size), # added
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, emb_size),
            # torch.nn.LayerNorm(emb_size, emb_size), # added
            torch.nn.LeakyReLU(),
        )

        # VARIABLE EMBEDDING
        self.var_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(var_nfeats),
            torch.nn.Linear(var_nfeats, emb_size),
            # torch.nn.LayerNorm(emb_size, emb_size), # added
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, emb_size),
            # torch.nn.LayerNorm(emb_size, emb_size), # added
            torch.nn.LeakyReLU(),
        )

        self.conv_v_to_c = BipartiteGraphConvolution(emb_size=emb_size, aggregator='mean')
        self.conv_c_to_v = BipartiteGraphConvolution(emb_size=emb_size, aggregator='mean')

        heads = []
        for _ in range(self.num_heads):
            head = []
            for _ in range(self.head_depth):
                head.append(torch.nn.Linear(emb_size, emb_size))
                head.append(torch.nn.LeakyReLU())
            head.append(torch.nn.Linear(emb_size, 1, bias=True))
            heads.append(torch.nn.Sequential(*head))
        self.heads_module = torch.nn.ModuleList(heads)

        self.init_model_parameters()
        self.to(device)


    def init_model_parameters(self, init_gnn_params=True, init_heads_params=True):

        def init_params(m):
            if isinstance(m, torch.nn.Linear):
                # weights
                if self.linear_weight_init is None:
                    pass
                elif self.linear_weight_init == 'uniform':
                    torch.nn.init.uniform_(m.weight, a=0.0, b=1.0)
                elif self.linear_weight_init == 'normal':
                    torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
                elif self.linear_weight_init == 'xavier_uniform':
                    torch.nn.init.xavier_uniform_(m.weight, gain=torch.nn.init.calculate_gain(self.activation))
                elif self.linear_weight_init == 'xavier_normal':
                    torch.nn.init.xavier_normal_(m.weight, gain=torch.nn.init.calculate_gain(self.activation))
                elif self.linear_weight_init == 'kaiming_uniform':
                    torch.nn.init.kaiming_uniform_(m.weight, nonlinearity=self.activation)
                elif self.linear_weight_init == 'kaiming_normal':
                    torch.nn.init.kaiming_normal_(m.weight, nonlinearity=self.activation)
                    # torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                else:
                    raise Exception(f'Unrecognised linear_weight_init {self.linear_weight_init}')

                # biases
                if m.bias is not None:
                    if self.linear_bias_init is None:
                        pass
                    elif self.linear_bias_init == 'zeros':
                        torch.nn.init.zeros_(m.bias)
                    elif self.linear_bias_init == 'uniform':
                        torch.nn.init.uniform_(m.bias)
                    elif self.linear_bias_init == 'normal':
                        torch.nn.init.normal_(m.bias)
                    else:
                        raise Exception(f'Unrecognised bias initialisation {self.linear_bias_init}')

            elif isinstance(m, torch.nn.LayerNorm):
                # weights
                if self.layernorm_weight_init is None:
                    pass
                elif self.layernorm_weight_init == 'normal':
                    torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    raise Exception(f'Unrecognised layernorm_weight_init {self.layernorm_weight_init}')

                # biases
                if self.layernorm_bias_init is None:
                    pass
                elif self.layernorm_bias_init == 'zeros':
                    torch.nn.init.zeros_(m.bias)
                elif self.layernorm_bias_init == 'normal':
                    torch.nn.init.normal_(m.bias)
                else:
                    raise Exception(f'Unrecognised layernorm_bias_init {self.layernorm_bias_init}')

        if init_gnn_params:
            # init base GNN params
            self.apply(init_params)

        if init_heads_params:
            # init head output params
            for h in self.heads_module:
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
        head_output = [self.heads_module[head](variable_features).squeeze(-1) for head in range(self.num_heads)]
        head_output = torch.stack(head_output, dim=0).mean(dim=0)

        return head_output


class Agent:
    def __init__(self, device, epsilon=0.1):
        self.net = BipartiteGCN(device=device, var_nfeats=24)
        self.epsilon = epsilon
        self.opt = torch.optim.Adam(self.net.parameters())

    def act(self, obs, action_set, deterministic=False):
        with torch.no_grad():
            preds = self.net(obs)[action_set.astype('int32')]

        # single observation
        if np.random.rand() < self.epsilon * (1 - deterministic):
            action = np.random.choice(action_set)
        else:
            action_idx = torch.argmax(preds)
            action = action_set[action_idx.item()]
        return action

    def update(self, obs_batch, act_batch, ret_batch):
        self.opt.zero_grad()
        loss = torch.tensor(0, requires_grad=True)
        # TODO use torch_geometric.data.Batch
        for obs, act, ret in zip(obs_batch, act_batch, ret_batch):
            pred = self.net(obs)[act]
            loss += ((pred - ret) ** 2) / len(obs_batch)
        loss.backward()
        self.opt.step()
        return loss.detach().cpu().item()

    def save(self, path, epoch_id):
        torch.save(self.net.state_dict(), path + f'/checkpoint_{epoch_id}.pkl')

    def load(self, path, epoch_id):
        self.net.load_state_dict(
            torch.load(path + f'/checkpoint_{epoch_id}.pkl', map_location=self.net.device)
        )

    def train(self):
        self.net.train()

    def eval(self):
        self.net.eval()
