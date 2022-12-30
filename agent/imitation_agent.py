import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric
from utils import pad_tensor
from env import make_tripartite
from utils import UnpackedBipartite, UnpackedTripartite


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
                 var_nfeats=19,
                 num_heads=1,
                 head_depth=1,
                 linear_weight_init=None,
                 linear_bias_init=None,
                 layernorm_weight_init=None,
                 layernorm_bias_init=None,
                 encode_possible_actions=True):
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

        self.encode_possible_actions = encode_possible_actions
        if encode_possible_actions:
            self.pos_act_emb = nn.Embedding(2, emb_size)

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
        constraint_features = obs.row_features
        edge_indices = obs.edge_index
        edge_features = obs.edge_attr
        variable_features = obs.variable_features

        # print(constraint_features.shape, edge_indices.shape, edge_features.shape, variable_features.shape)
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

        # First step: linear embedding layers to a common dimension (64)
        constraint_features = self.cons_embedding(constraint_features)
        assert variable_features.shape[1] == self.var_nfeats
        variable_features = self.var_embedding(variable_features)

        if self.encode_possible_actions:
            pos = torch.zeros_like(variable_features[:, 0])
            pos[obs.candidates] = 1
            pos_encodes = self.pos_act_emb(pos.long())
            variable_features += pos_encodes

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

        return head_output[obs.candidates]


class TripartiteGCN(torch.nn.Module):
    def __init__(self,
                 device,
                 emb_size=64,
                 cons_nfeats=5,
                 edge_nfeats=1,
                 var_nfeats=19,
                 cut_nfeats=2,
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
        self.cut_nfeats = cut_nfeats
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

        # CUT EMBEDDING
        self.cut_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(cut_nfeats),
            torch.nn.Linear(cut_nfeats, emb_size),
            # torch.nn.LayerNorm(emb_size, emb_size), # added
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, emb_size),
            # torch.nn.LayerNorm(emb_size, emb_size), # added
            torch.nn.LeakyReLU(),
        )

        self.conv_v_to_c = BipartiteGraphConvolution(emb_size=emb_size, aggregator='mean')
        self.conv_c_to_v = BipartiteGraphConvolution(emb_size=emb_size, aggregator='mean')
        self.conv_cut_to_v = BipartiteGraphConvolution(emb_size=emb_size, aggregator='mean')
        self.conv_v_to_cut = BipartiteGraphConvolution(emb_size=emb_size, aggregator='mean')
        self.conv_cut_to_c = BipartiteGraphConvolution(emb_size=emb_size, aggregator='mean')
        self.conv_c_to_cut = BipartiteGraphConvolution(emb_size=emb_size, aggregator='mean')

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
        constraint_features = obs.row_features
        variable_features = obs.variable_features
        cut_features = obs.cut_features

        edge_indices = obs.edge_index
        cut_row_edge_indices = obs.cut_row_edge_index
        cut_col_edge_indices = obs.cut_col_edge_index

        edge_features = obs.edge_features
        cut_row_edge_features = obs.cut_row_edge_features
        cut_col_edge_features = obs.cut_col_edge_features

        # print(constraint_features.shape, edge_indices.shape, edge_features.shape, variable_features.shape)
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)
        reversed_cut_col_edge_indices = torch.stack([cut_col_edge_indices[1], cut_col_edge_indices[0]], dim=0)
        reversed_cut_row_edge_indices = torch.stack([cut_row_edge_indices[1], cut_row_edge_indices[0]], dim=0)

        # First step: linear embedding layers to a common dimension (64)
        constraint_features = self.cons_embedding(constraint_features)
        #print(variable_features.shape, torch.randn(1).item())
        assert variable_features.shape[1] == self.var_nfeats
        variable_features = self.var_embedding(variable_features)
        cut_features = self.cut_embedding(cut_features)

        # Two half convolutions (message passing round)
        constraint_features = self.conv_v_to_c(
            variable_features, reversed_edge_indices, edge_features, constraint_features
        )
        variable_features = self.conv_c_to_v(
            constraint_features, edge_indices, edge_features, variable_features
        )
        variable_features = self.conv_cut_to_v(
            cut_features, cut_col_edge_indices, cut_col_edge_features, variable_features
        )
        cut_features = self.conv_v_to_cut(
            variable_features, reversed_cut_col_edge_indices, cut_col_edge_features, cut_features,
        )
        constraint_features = self.conv_cut_to_c(
            cut_features, cut_row_edge_indices, cut_row_edge_features, constraint_features
        )
        cut_features = self.conv_c_to_cut(
            constraint_features, reversed_cut_row_edge_indices, cut_row_edge_features, cut_features
        )

        # get output for each head
        head_output = [self.heads_module[head](cut_features).squeeze(-1) for head in range(self.num_heads)]
        head_output = torch.stack(head_output, dim=0).mean(dim=0)

        return head_output


class ImitationAgent:
    def __init__(self, device, observation_format='tripartite', target='expert_actions', loss_function='cross_entropy',
                 encode_possible_actions=True):
        self.device = device
        assert observation_format in ['bipartite', 'tripartite']
        self.observation_format = observation_format
        if observation_format == 'bipartite':
            self.net = BipartiteGCN(device=device,
                                    encode_possible_actions=encode_possible_actions)
        elif observation_format == 'tripartite':
            self.net = TripartiteGCN(device=device)
        self.opt = torch.optim.Adam(self.net.parameters())

        assert target in ['expert_actions', 'expert_scores'], 'target not implemented'
        self.target = target

        assert loss_function in ['cross_entropy', 'mse'], 'loss function not implemented'
        if loss_function == 'cross_entropy':
            self.loss = CrossEntropy()
        elif loss_function == 'mse':
            self.loss = MeanSquaredError()

        self.device = device

    def act(self, obs, action_set, deterministic, env):
        obs = make_tripartite(env, obs, action_set)
        if self.observation_format == 'bipartite':
            obs = UnpackedBipartite(obs, action_set, self.device)
        elif self.observation_format == 'tripartite':
            obs = UnpackedTripartite(obs, self.device)

        with torch.no_grad():
            preds = self.net(obs)
        action_idx = torch.argmax(preds)
        action = action_set[action_idx.item()]
        return action, None

    def update(self, obs):
        self.opt.zero_grad()
        pred = self.net(obs)

        if self.target == 'expert_actions':
            target = obs.candidate_choices
        elif self.target == 'expert_scores':
            target = obs.candidate_scores

        loss = self.loss(pred, target, obs.num_candidates)
        loss.backward()
        self.opt.step()
        return loss.detach().cpu().item()

    def validate(self, obs):
        self.opt.zero_grad()
        with torch.no_grad():
            pred = self.net(obs)

            if self.target == 'expert_actions':
                target = obs.candidate_choices
            elif self.target == 'expert_scores':
                target = obs.candidate_scores

            loss = self.loss(pred, target, obs.num_candidates)
        return loss.detach().cpu().item()

    def save(self, path, epoch_id):
        torch.save(self.net.state_dict(), path + f'/checkpoint_{epoch_id}.pkl')

    def load(self, path):
        self.net.load_state_dict(torch.load(path, map_location=self.device))

    def train(self):
        self.net.train()

    def eval(self):
        self.net.eval()


class MeanSquaredError(nn.Module):
    def __init__(self, reduction='mean', norm_target=False):
        super().__init__()
        self.reduction = reduction
        self.norm_target = norm_target

    def forward(self, _input, target, num_candidates=None):
        if self.norm_target:
            with torch.no_grad():
                target = target/target.max()

        if num_candidates is not None:
            _input = pad_tensor(_input, num_candidates)
            target = pad_tensor(target, num_candidates)

        if self.reduction == 'clip':
            reduction = 'none'
            loss = F.mse_loss(_input, target, reduction=reduction)
            loss = torch.clip(loss, max=1)
            return torch.mean(loss)
        else:
            return F.mse_loss(_input, target, reduction=self.reduction)


class CrossEntropy(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, _input, target, num_candidates=None):
        if num_candidates is not None:
            _input = pad_tensor(_input, num_candidates)
        return F.cross_entropy(_input, target, reduction=self.reduction)
