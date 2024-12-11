# -*- encoding: utf-8 -*-
'''
@File    :   model.py
@Time    :   2021/02/19 21:10:00
@Author  :   Fei gao
@Contact :   feig@mail.bnu.edu.cn
BNU, Beijing, China
'''
import copy
import math

import torch
from torch import nn, optim

import torch.nn.functional as F
# from torch.nn.modules.loss import BCEWithLogitsLoss
from torch_geometric.utils import softmax
from torch_scatter import scatter
import numpy as np
from Layer_mv100k import LightGCN
from utils import utilsaf

# from utils.utilsaf import BPRLoss
# from train import args


# from models.layers import StructuralAttentionLayer, TemporalAttentionLayer
# from utils.utilities import fixed_unigram_candidate_sampler
def fixed_unigram_candidate_sampler(true_clasees,
                                    num_true,
                                    num_sampled,
                                    unique,
                                    distortion,
                                    unigrams):
    # TODO: implementate distortion to unigrams
    assert true_clasees.shape[1] == num_true
    samples = []
    for i in range(true_clasees.shape[0]):
        dist = copy.deepcopy(unigrams)
        candidate = list(range(len(dist)))
        taboo = true_clasees[i].cpu().tolist()
        for tabo in sorted(taboo, reverse=True):
            candidate.remove(tabo)
            dist.pop(tabo)
        sample = np.random.choice(candidate, size=num_sampled, replace=unique, p=dist / np.sum(dist))
        samples.append(sample)
    return samples


# class PairWiseModel(BasicModel):
#     def __init__(self):
#         super(PairWiseModel, self).__init__()
#
#     def bpr_loss(self, users, pos, neg):
#         """
#         Parameters:
#             users: users list
#             pos: positive items for corresponding users
#             neg: negative items for corresponding users
#         Return:
#             (log-loss, l2-loss)
#         """
#         raise NotImplementedError

class Positional_Encoding(nn.Module):
    def __init__(self, d_model):
        super(Positional_Encoding, self).__init__()
        self.d_model = d_model

    def forward(self, seq_len, embedding_dim):
        positional_encoding = np.zeros((seq_len, embedding_dim))
        for pos in range(positional_encoding.shape[0]):
            for i in range(positional_encoding.shape[1]):
                positional_encoding[pos][i] = math.sin(pos/(10000**(2*i/self.d_model))) \
                    if i % 2 == 0 else math.cos(pos/(10000**(2*i/self.d_model)))
        return torch.from_numpy(positional_encoding)


class StructuralAttentionLayer(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 n_heads,
                 attn_drop,
                 ffd_drop,
                 residual):
        super(StructuralAttentionLayer, self).__init__()
        self.out_dim = output_dim // n_heads
        self.n_heads = n_heads
        self.act = nn.ELU()

        self.lin = nn.Linear(input_dim, n_heads * self.out_dim, bias=False)
        self.att_l = nn.Parameter(torch.Tensor(1, n_heads, self.out_dim))
        self.att_r = nn.Parameter(torch.Tensor(1, n_heads, self.out_dim))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

        self.attn_drop = nn.Dropout(attn_drop)
        self.ffd_drop = nn.Dropout(ffd_drop)

        self.residual = residual
        if self.residual:
            self.lin_residual = nn.Linear(input_dim, n_heads * self.out_dim, bias=False)

        self.xavier_init()

    def forward(self, graph):
        graph = copy.deepcopy(graph)
        edge_index = graph.edge_index
        edge_weight = graph.edge_weight.reshape(-1, 1)
        H, C = self.n_heads, self.out_dim
        x = self.lin(graph.x).view(-1, H, C)  # [N, heads, out_dim]
        # attention
        alpha_l = (x * self.att_l).sum(dim=-1).squeeze()  # [N, heads]
        alpha_r = (x * self.att_r).sum(dim=-1).squeeze()
        alpha_l = alpha_l[edge_index[0]]  # [num_edges, heads]
        alpha_r = alpha_r[edge_index[1]]
        alpha = alpha_r + alpha_l
        alpha = edge_weight * alpha
        alpha = self.leaky_relu(alpha)
        coefficients = softmax(alpha, edge_index[1])  # [num_edges, heads]

        # dropout
        if self.training:
            coefficients = self.attn_drop(coefficients)
            x = self.ffd_drop(x)
        x_j = x[edge_index[0]]  # [num_edges, heads, out_dim]

        # output
        out = self.act(scatter(x_j * coefficients[:, :, None], edge_index[1], dim=0, reduce="sum"))
        out = out.reshape(-1, self.n_heads * self.out_dim)  # [num_nodes, output_dim]
        if self.residual:
            out = out + self.lin_residual(graph.x)
        graph.x = out
        return graph

    def xavier_init(self):
        nn.init.xavier_uniform_(self.att_l)
        nn.init.xavier_uniform_(self.att_r)


class TemporalAttentionLayer(nn.Module):
    def __init__(self,
                 input_dim,
                 n_heads,
                 num_time_steps_,
                 attn_drop,
                 residual,
                 position_embeddings):
        super(TemporalAttentionLayer, self).__init__()
        self.n_heads = n_heads
        self.num_time_steps = num_time_steps_-2
        self.residual = residual

        # define weights
        self.position_embeddings =  position_embeddings#nn.Parameter(torch.Tensor(self.num_time_steps, input_dim))
        self.Q_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))  # 128*128
        self.K_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.V_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        # ff
        self.lin = nn.Linear(input_dim, input_dim, bias=True)
        # dropout
        self.attn_dp = nn.Dropout(attn_drop)
        self.xavier_init()

    def forward(self, inputs):
        """In:  attn_outputs (of StructuralAttentionLayer at each snapshot):= [N, T, F]"""
        # 1: Add position embeddings to input

        # position_inputs = torch.arange(0, self.num_time_steps).reshape(1, -1).repeat(inputs.shape[0], 1).long().to(
        #     inputs.device)

        temporal_inputs = inputs# + self.position_embeddings[position_inputs]  # [N, T, F]

        # 2: Query, Key based multi-head self attention.
        q = torch.tensordot(temporal_inputs, self.Q_embedding_weights, dims=([2], [0]))  # [N, T, F]
        k = torch.tensordot(temporal_inputs, self.K_embedding_weights, dims=([2], [0]))  # [N, T, F]
        v = torch.tensordot(temporal_inputs, self.V_embedding_weights, dims=([2], [0]))  # [N, T, F]

        # 3: Split, concat and scale.
        split_size = int(q.shape[-1] / 1)#self.n_heads
        q_ = torch.cat(torch.split(q, split_size_or_sections=split_size, dim=2), dim=0)  # [hN, T, F/h]
        k_ = torch.cat(torch.split(k, split_size_or_sections=split_size, dim=2), dim=0)  # [hN, T, F/h]
        v_ = torch.cat(torch.split(v, split_size_or_sections=split_size, dim=2), dim=0)  # [hN, T, F/h]
        # print(sum(sum(sum(torch.isnan(q_)))))
        # print(sum(sum(sum(torch.isnan(k_)))))
        # print(sum(sum(sum(torch.isnan(v_)))))
        outputs = torch.matmul(q_, k_.permute(0, 2, 1))  # [hN, T, T]
        outputs = outputs / (self.num_time_steps ** 0.5)
        # 4: Masked (causal) softmax to compute attention weights.
        diag_val = torch.ones_like(outputs[0])
        tril = torch.tril(diag_val)
        masks = tril[None, :, :].repeat(outputs.shape[0], 1, 1)  # [h*N, T, T]
        padding = torch.ones_like(masks) * (-2 ** 32 + 1)
        outputs = torch.where(masks == 0, padding, outputs)
        outputs = F.softmax(outputs, dim=2)
        self.attn_wts_all = outputs  # [255168, 7, 7] [h*N, T, T]

        # 5: Dropout on attention weights.
        if self.training:
            outputs = self.attn_dp(outputs)
        outputs = torch.matmul(outputs, v_)  # [hN, T, F/h]
        # outputs = torch.cat(torch.split(outputs, split_size_or_sections=int(outputs.shape[0] / self.n_heads),dim=0), dim=2)
        # 6: Feedforward and residual
        outputs = self.feedforward(outputs)
        if self.residual:# 残差边
            outputs = outputs + temporal_inputs
        return outputs  # [14528+1420，t, dim]

    def feedforward(self, inputs):
        outputs = F.relu(self.lin(inputs))
        return outputs + inputs

    def xavier_init(self):
        nn.init.xavier_uniform_(self.Q_embedding_weights)
        nn.init.xavier_uniform_(self.K_embedding_weights)
        nn.init.xavier_uniform_(self.V_embedding_weights)


class DySAT(nn.Module):
    def __init__(self,
                 args,
                 num_features,
                 time_length,
                 datasetloder,
                 graphs):
        """[summary]

        Args:
            args ([type]): [description]
            time_length (int): Total timesteps in dataset.
        """
        super(DySAT, self).__init__()
        self.args = args
        if args.window < 0:
            self.num_time_steps = time_length
        else:
            self.num_time_steps = min(time_length, args.window + 1)  # window = 0 => only self.
        self.num_features = num_features
        self.datasetloder = datasetloder
        self.graphs = graphs
        self.loss_graph = []
        # 位置编码
        self.positional_encodings = self.get_positional_encoding(seq_len=self.num_time_steps, d_model=args.latent_dim).cuda()
        # self.position_embeddings = torch.nn.Embedding(num_embeddings=self.num_time_steps, embedding_dim=args.latent_dim)
        # nn.init.normal_(self.position_embeddings.weight, std=0.1)

        self.aflayer_list = LightGCN(args=self.args,
                                       datasetloder=self.datasetloder,
                                       graphs=self.graphs,
                                       position_embeddings=self.positional_encodings[:-2])

        self.temporal_head_config = list(map(int, args.temporal_head_config.split(",")))
        self.temporal_layer_config = list(map(int, args.temporal_layer_dim.split(",")))
        self.spatial_drop = args.spatial_drop
        self.temporal_drop = args.temporal_drop
        self.temporal_attn = self.build_model()

        self.norm = nn.LayerNorm(normalized_shape=self.args.latent_dim, eps=1e-6)
        self.linear = nn.Linear(in_features=args.latent_dim, out_features=args.latent_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.f = nn.Sigmoid()

    def forward(self):

        structural_out = self.aflayer_list.computer()
        structural_outputs = [g.view(-1, 1, self.args.latent_dim) for g in structural_out]  # list of [Ni, 1, F]

        structural_outputs_padded = torch.cat(structural_outputs, dim=1)  # [N, T, F] = [15,948, 9, 64]

        self.structural_outputs_padded = structural_outputs_padded.permute(1, 0, 2) + \
                                    self.positional_encodings[:-2].unsqueeze(1).expand(-1, structural_outputs_padded.shape[0], -1)
        #######################################################################
        self.temporal_out = self.temporal_attn(structural_outputs_padded)[:, -1, :].squeeze()
        #######################################################################
        loss_total = 0

        loss_total = loss_total + self.batch_loss(self.datasetloder[-2], self.temporal_out)

        for t in range(self.num_time_steps - 2):
            loss_total = loss_total + self.batch_loss(self.datasetloder[t], self.structural_outputs_padded[t, :, :])

        return loss_total

    def get_positional_encoding(self, seq_len, d_model):
        # 初始化位置编码矩阵
        positional_encoding = torch.zeros(seq_len, d_model)

        # 计算位置编码
        for pos in range(seq_len):
            for i in range(0, d_model, 2):
                positional_encoding[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                if i + 1 < d_model:
                    positional_encoding[pos, i + 1] = math.cos(pos / (10000 ** ((2 * i) / d_model)))

        return positional_encoding

    def batch_loss(self, graph, g_embed):
        total_batch = 0
        aver_loss = 0.

        S = utilsaf.UniformSample_original(graph)
        users = torch.Tensor(S[:, 0]).long()
        posItems = torch.Tensor(S[:, 1]).long()
        negItems = torch.Tensor(S[:, 2]).long()

        users = users.to(self.args.device)
        posItems = posItems.to(self.args.device)
        negItems = negItems.to(self.args.device)
        users, posItems, negItems = utilsaf.shuffle(users, posItems, negItems)
        total_batch += len(users) // self.args.bpr_batch_size + 1

        for (batch_i,
             (batch_users,
              batch_pos,
              batch_neg)) in enumerate(utilsaf.minibatch(users,
                                                         posItems,
                                                         negItems,
                                                         batch_size=self.args.bpr_batch_size)):

            loss, reg_loss = self.bpr_loss(batch_users, batch_pos, batch_neg, g_embed)  ## 输出12个Graph的
            reg_loss = reg_loss * self.args.weight_decay
            loss = loss + reg_loss
            aver_loss = aver_loss + loss
        self.loss_graph.append(aver_loss.cpu().item())

        if len(self.loss_graph) == self.num_time_steps-2:
            print('Every Graph-Loss List:', self.loss_graph)
            self.loss_graph = []

        return aver_loss

    def getEmbedding(self, users, pos_items, neg_items, g_embed):
        # 单个图的
        all_users, all_items = torch.split(g_embed, [self.aflayer_list.num_users, self.aflayer_list.num_items])
        # all_users, all_items = self.computer() # 得到的是列表 12*14528*64
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = all_users[users]
        pos_emb_ego = all_items[pos_items]
        neg_emb_ego = all_items[neg_items]

        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, users, pos, neg, g_embed):

        # for g_embed in g_embeds:
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long(), g_embed)
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        return loss, reg_loss

    def build_model(self):
        input_dim = self.num_features

        temporal_attention_layers = nn.Sequential()
        for i in range(len(self.temporal_layer_config)):
            layer = TemporalAttentionLayer(input_dim=input_dim,
                                           n_heads=self.temporal_head_config[i],
                                           num_time_steps_=self.num_time_steps,
                                           attn_drop=self.temporal_drop,
                                           residual=self.args.residual,
                                           position_embeddings=self.positional_encodings[:-2])
            temporal_attention_layers.add_module(name="temporal_layer_{}".format(i), module=layer)
            input_dim = self.temporal_layer_config[i]

        return temporal_attention_layers

    def get_pre_embedd(self, t):

        in_test = self.structural_outputs_padded[-1, :, :] + self.temporal_out # 短期兴趣+长期兴趣表示
        # pred_embedd = self.transformer(in_test, in_test)
        pred_embedd = in_test

        pred_embedd_lin = pred_embedd.view(-1, 64)
        pred_embedd_lin = self.linear(pred_embedd_lin)
        pred_embedd_out = self.softmax(pred_embedd_lin)
        return pred_embedd_out

    def getUsersRating(self, users,t):  # 求users和item的偏好评级
        pred_embedd_out = self.get_pre_embedd(t)
        all_users, all_items =torch.split(pred_embedd_out, [self.aflayer_list.num_users, self.aflayer_list.num_items])
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating
