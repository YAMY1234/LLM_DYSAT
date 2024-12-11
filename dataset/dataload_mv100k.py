
import os
from os.path import join
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix
import scipy.sparse as sp
# import world
# from world import cprint
from time import time
from glob import glob
import pandas as pd

class BasicDataset(Dataset):
    def __init__(self):
        print("init dataset")

    @property
    def n_users(self):
        raise NotImplementedError

    @property
    def m_items(self):
        raise NotImplementedError

    @property
    def trainDataSize(self):
        raise NotImplementedError

    @property
    def testDict(self):
        raise NotImplementedError

    @property
    def allPos(self):
        raise NotImplementedError

    def getUserItemFeedback(self, users, items):
        raise NotImplementedError

    def getUserPosItems(self, users):
        raise NotImplementedError

    def getUserNegItems(self, users):
        """
        not necessary for large dataset
        it's stupid to return all neg items in super large dataset
        """
        raise NotImplementedError

    def getSparseGraph(self):
        """
        build a graph in torch.sparse.IntTensor.
        Details in NGCF's matrix form
        A = 
            |I,   R|
            |R^T, I|
        """
        raise NotImplementedError

device = torch.device('cuda')
# ----------------------------------------------------------------------------------------------------------------------
class Loader(BasicDataset):
    """
    Dataset type for pytorch \n
    Incldue graph information
    movie100k dataset
    user:age | gender | occupation
    item:genre
    """
    def __init__(self, graph, idd, path):
        # train or test
        self.path = path
        train_file = path + '/train_.txt'
        test_file = path + '/test_.txt'

        self.graph = graph
        self.split = False
        self.folds = 100 #config['A_n_fold']  # 用于拆分大型的adj矩阵  100
        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train']  # 0
        self.n_user = 943
        self.m_item = 1682

        self.n_att1 = 0  # 数值最终归为节点  用于网络
        self.n_att2 = 0
        self.n_att3 = 0
        self.m_att1 = 0
        ##################################################


        files = glob('E:/项目/社交推荐/dataset/kh/*.txt')


        for filex in files:
            if '_m_' in filex and str(idd) in filex:
                self.m_att = np.loadtxt(filex, dtype=float, delimiter=',', encoding='utf-8')

            if '_u_' in filex and str(idd) in filex:
                self.u_att = np.loadtxt(filex, dtype=float, delimiter=',', encoding='utf-8')[: 943]

        ##################################################

        self.traindataSize = 0
        self.testDataSize = 0
        self.userAtt1DataSize = 0
        self.userAtt2DataSize = 0
        self.userAtt3DataSize = 0
        self.itemAtt1DataSize = 0

        self.trainUser = np.loadtxt(train_file, dtype=int, delimiter=',', skiprows=0, usecols=(0, 1),
                                unpack=False,encoding="utf_8")
        self.testItem = np.loadtxt(test_file, dtype=int, delimiter=',', skiprows=0, usecols=(0, 1),
                                unpack=False, encoding="utf_8")

        self.traindataSize = len(self.trainUser)




        self.Graph = None
        self.Graph_att = None

        # (users,items), bipartite graph  二分图  构建稀疏矩阵  形成二分图
        # self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),  # 生成一个用户项交互矩阵R
        #                               shape=(self.n_user, self.m_item))
        self.UserItemNet = graph # (943, 1682)
        # (users, att1s)

        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()  # 得到保存每个图的user的度的一个行向量
        # sum(axis=1)将一个矩阵的每一行向量相加，squeeze()从数组的形状中删除单维度条目，即把shape中为1的维度去掉
        self.users_D[self.users_D == 0.] = 1  # 对与没有交互item的user，度为0，将他们的度设置为1    加自环
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()  # 得到保存item的度的一个行向量
        self.items_D[self.items_D == 0.] = 1.  # 加自环
        # pre-calculate  预计算
        self._allPos = self.getUserPosItems(list(range(self.n_user)))  # 没有与user交互的item
        self.__testDict = self.__build_test()  # 测试集的字典形式
        # print(f"{path} is ready to go")
    @property
    def n_users(self):
        return self.n_user
    @property
    def m_items(self):
        return self.m_item
    @property
    def n_att1s(self):
        return self.n_att1
    @property
    def n_att2s(self):
        return self.n_att2
    @property
    def n_att3s(self):
        return self.n_att3
    @property
    def m_att1s(self):
        return self.m_att1

    @property
    def UserAttNets(self):
        return self.UserAttNet

    @property
    def ItemAttNets(self):
        return self.ItemAttNet

    @property
    def trainDataSize(self):
        return self.traindataSize

    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    def _split_A_hat(self, A):
        A_fold = []
        fold_len = (self.n_users + self.m_items) // self.folds
        for i_fold in range(self.folds):
            start = i_fold * fold_len
            if i_fold == self.folds - 1:
                end = self.n_users + self.m_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(device))
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):  # 将矩阵转化成张量
        coo = X.tocoo().astype(np.float32)  # 返回矩阵的coo格式：仅存储非0数据的行、列、值；astype()转换numpy数组的数据类型
        row = torch.Tensor(coo.row).long()  # coo的行
        col = torch.Tensor(coo.col).long()  # coo的列
        index = torch.stack([row, col])  # 将两个一维的拼接成一个二维的
        data = torch.FloatTensor(coo.data)  # coo的值
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def getSparseGraph_att(self, t):
        print("loading adjacency matrix")
        if self.Graph_att is None:
            try:
                pre_adj_mat = sp.load_npz(
                    self.path + '/sp_npz/s_pre_att_adj_mat_{}.npz'.format(t))  # 查看形状 70742x70742 sparse matrix  1132025 stored
                print("successfully loaded...")
                norm_adj = pre_adj_mat
            except:
                print("generating adjacency matrix")
                s = time()  # 返回当前时间的时间戳（1970纪元后经过的浮点秒数）。
                adj_mat = sp.dok_matrix((2 * (self.n_users + self.m_items), 2 * (self.n_users + self.m_items)),
                                        dtype=np.float32)  # 生成稀疏矩阵，得到一个全为0的矩阵
                adj_mat = adj_mat.tolil()  # 将此矩阵转换为列表格式。当copy=False时，数据/索引可在该矩阵和生成的lil_矩阵之间共享。
                Ru = np.identity(self.n_users) # 用户u
                Ri = np.identity(self.m_items) # 物品i
                R = self.UserItemNet.tolil()
                adj_mat[:self.n_users, 2 * self.n_users + self.m_items:] = R
                adj_mat[2 * self.n_users + self.m_items:, :self.n_users] = R.T
                adj_mat[:self.n_users, self.n_users + self.m_items:2 * self.n_users + self.m_items] = Ru
                adj_mat[2 * self.n_users + self.m_items:, self.n_users:self.n_users + self.m_items] = Ri

                adj_mat = adj_mat.todok()  # 将此矩阵转换为键值字典格式   得到邻接矩阵A
                # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])

                rowsum = np.array(adj_mat.sum(axis=1))  # 对邻接矩阵A行求和，得到度
                d_inv = np.power(rowsum, -0.5).flatten()  # flatten()返回一个折叠成一维的数组，再求-0.5次方
                d_inv[np.isinf(d_inv)] = 0.  # np.isinf 判断元素是否为无穷大，是则为True，否则为False，返回形状相同的布尔数组,将无穷大（度为0的行）的值换成0
                d_mat = sp.diags(d_inv)  # 生成对角的度矩阵D^-2

                norm_adj = d_mat.dot(adj_mat)  # dot矩阵乘积 D^-2 * A
                d_mat = d_mat.tolil()
                d_mat[self.n_users:self.n_users + self.m_items, self.n_users:self.n_users + self.m_items] = Ri
                d_mat[self.n_users + self.m_items:2 * self.n_users + self.m_items,
                self.n_users + self.m_items:2 * self.n_users + self.m_items] = Ru
                d_mat = d_mat.todok()
                norm_adj = norm_adj.dot(d_mat)  # D^-2 * A * D'
                norm_adj = norm_adj.tocsr()  # 以压缩稀疏行格式返回此矩阵的副本
                end = time()
                print(f"costing {end - s}s, saved norm_mat...")
                sp.save_npz(self.path + '/sp_npz/s_pre_att_adj_mat_{}.npz'.format(t), norm_adj)

            if self.split == True:
                self.Graph_att = self._split_A_hat(norm_adj)
                print("done split matrix")
            else:
                self.Graph_att = self._convert_sp_mat_to_sp_tensor(norm_adj)  # size=(70742, 70742), nnz=1132025
                self.Graph_att = self.Graph_att.coalesce().to(device)
                print("don't split the matrix")
        return self.Graph_att

    def getSparseGraph(self, t):
        print("loading adjacency matrix")
        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz(self.path + '/sp_npz/s_pre_adj_mat_{}.npz'.format(t))  # 35371x35371 sparse matrix user-book adj
                print("successfully loaded...")
                norm_adj = pre_adj_mat
            except:
                print("generating adjacenc matrix")
                s = time()  # 返回当前时间的时间戳（1970纪元后经过的浮点秒数）。
                adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items),
                                        dtype=np.float32)  # 生成稀疏矩阵，得到一个全为0的矩阵
                adj_mat = adj_mat.tolil()  # 将此矩阵转换为列表格式。当copy=False时，数据/索引可在该矩阵和生成的lil_矩阵之间共享。
                R = self.UserItemNet.tolil()
                adj_mat[:self.n_users, self.n_users:] = R
                adj_mat[self.n_users:, :self.n_users] = R.T
                adj_mat = adj_mat.todok()  # 将此矩阵转换为键值字典格式   得到邻接矩阵A
                # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])

                rowsum = np.array(adj_mat.sum(axis=1))  # 对邻接矩阵A行求和，得到度
                d_inv = np.power(rowsum, -0.5).flatten()  # flatten()返回一个折叠成一维的数组，再求-0.5次方
                d_inv[np.isinf(d_inv)] = 0.  # np.isinf 判断元素是否为无穷大，是则为True，否则为False，返回形状相同的布尔数组-----将无穷大（度为0的行）的值换成0
                d_mat = sp.diags(d_inv)  # 生成对角的度矩阵D^-2

                norm_adj = d_mat.dot(adj_mat)  # dot矩阵乘积 D^-2 * A
                norm_adj = norm_adj.dot(d_mat)  # D^-2 * A * D^-2
                norm_adj = norm_adj.tocsr()  # 以压缩稀疏行格式返回此矩阵的副本
                end = time()
                print(f"costing {end - s}s, saved norm_mat...")
                sp.save_npz(self.path + '/sp_npz/s_pre_adj_mat_{}.npz'.format(t), norm_adj)

            if self.split == True:
                self.Graph = self._split_A_hat(norm_adj)
                print("done split matrix")
            else:
                self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)  # size=(35371, 35371), nnz=1096654
                self.Graph = self.Graph.coalesce().to(device)
                print("don't split the matrix")
        return self.Graph

    def __build_test(self):  # 把测试集转变成字典形式
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        # for i, item in enumerate(self.testItem):
        #     user = self.testUser[i] #
        #     if test_data.get(user): #
        #         test_data[user].append(item)
        #     else:
        #         test_data[user] = [item]
        # return test_data

    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserItemNet[users, items])  w
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    def getUserPosItems(self, users):  # 参数users：list(range(self.n_user))  返回user交互的item
        posItems = []  # 将交互过的item挑选出来   作为正样本
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems

    def getUserNegItems(self, users):
        negItems = []
        for user in users:
            negItems.append(self.allNeg[user])
        return negItems
