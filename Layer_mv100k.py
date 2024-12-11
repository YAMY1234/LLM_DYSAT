
import torch
from dataset.dataload_mv100k import BasicDataset
from torch import nn
import numpy as np
import torch.nn.functional as F

class AttentionFusion(nn.Module):
    def __init__(self, feature_dim):
        super(AttentionFusion, self).__init__()
        self.feature_dim = feature_dim
        # 可学习的映射函数，用于计算注意力得分
        self.query_fc = nn.Linear(feature_dim, feature_dim)
        self.key_fc = nn.Linear(feature_dim, feature_dim)
        self.value_fc = nn.Linear(feature_dim, feature_dim)

    def forward(self, a, b):
        # a, b: 输入特征，假设其形状为 (batch_size, seq_len, feature_dim)
        # 在这个例子中，seq_len = 1000, feature_dim = 64

        # 计算查询（Query）、键（Key）和值（Value）
        query = self.query_fc(a)  # shape: (batch_size, seq_len, feature_dim)
        key = self.key_fc(b)  # shape: (batch_size, seq_len, feature_dim)
        value = self.value_fc(b)  # shape: (batch_size, seq_len, feature_dim)

        # 计算注意力得分
        attention_scores = torch.matmul(query, key.transpose(-1, -2))  # shape: (batch_size, seq_len, seq_len)
        attention_weights = F.softmax(attention_scores, dim=-1)  # shape: (batch_size, seq_len, seq_len)

        # 使用注意力权重对值进行加权求和
        weighted_sum = torch.matmul(attention_weights, value)  # shape: (batch_size, seq_len, feature_dim)

        return weighted_sum


class LightGCN(nn.Module):

    def __init__(self, args: dict, datasetloder: BasicDataset, graphs: list, position_embeddings):
        super(LightGCN, self).__init__()
        self.args = args
        self.graphs = graphs
        self.dataset: BasicDataset = datasetloder
        self.position_embeddings = position_embeddings
        self.__init_weight()
        self.attention_fusion = AttentionFusion(64)

        self.linear_txt = nn.Linear(768, 64)
        self.u_att_list = [torch.FloatTensor(dt.u_att).to("cuda:0") for dt in self.dataset]
        self.m_att_list = [torch.FloatTensor(dt.m_att).to("cuda:0") for dt in self.dataset]

    def __init_weight(self):
        self.num_users = self.dataset[-2].n_users
        self.num_items = self.dataset[-2].m_items
        self.num_useratt1 = self.dataset[-2].n_att1s
        self.num_useratt2 = self.dataset[-2].n_att2s
        self.num_useratt3 = self.dataset[-2].n_att3s
        self.num_itematt1 = self.dataset[-2].m_att1s

        self.num_time_steps = self.args.time_steps
        self.latent_dim = self.args.latent_dim  # 模型嵌入的维度
        self.n_layers = self.args.layers  # 层数 3
        self.keep_prob = self.args.keep_prob  # BRP损失的batch大小  0.6
        self.A_split = self.args.A_split  # False


        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)

        self.Graph = [dataset_in.getSparseGraph(t) for t, dataset_in in enumerate(self.dataset)]

        print(f"lgn is already to go(dropout:{self.args.spatial_drop})")

    def __dropout_x(self, x, keep_prob):  # unuse
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):  # unuse
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def normalize(self, tensorss):

        tensorss = tensorss.float()

        min_val = torch.min(tensorss)
        max_val = torch.max(tensorss)

        normalized_tensor = (tensorss - min_val) / (max_val - min_val)
        return normalized_tensor

    def computer(self):
        light_outs = []

        lamda = 0.6
        for t in range(len(self.Graph)-2):
            embs = []

            users_emb = self.embedding_user.weight
            items_emb = self.embedding_item.weight

            users_emb_position = self.position_embeddings[t] * lamda + (1-lamda) * users_emb
            items_emb_position = self.position_embeddings[t] * lamda + (1-lamda) * items_emb
            # torch.isnan()

            all_emb = torch.cat([users_emb_position, items_emb_position])  # 2.考虑将属性进行更高层的卷积处理
            txt_emb = torch.cat([self.linear_txt(self.u_att_list[t]), self.linear_txt(self.m_att_list[t])])  # 2.考虑将属性进行更高层的卷积处理
            embs.append(all_emb)
            for i in range(self.n_layers - 1):
                all_emb = torch.sparse.mm(self.Graph[t], all_emb)  # 3.考虑将属性进行更高层的卷积处理
                embs.append(all_emb)
            embs = torch.stack(embs, dim=1)  # torch.stack()在dim维度上连接若干个张量
            light_out = self.normalize(torch.mean(embs, dim=1))  # torch.Size([15948, 4, 64])   # 对所有层的嵌入求均值

            end_out = self.attention_fusion(torch.unsqueeze(txt_emb, 0), torch.unsqueeze(light_out, 0)) + light_out

            light_outs.append(end_out)

        return light_outs


    def normalization(self, matrix):  # 1
        zero_vec = -9e15 * torch.ones_like(matrix)
        attention = torch.where(matrix != 0, matrix, zero_vec)
        softmax = torch.nn.Softmax(dim=1)
        return softmax(attention)
