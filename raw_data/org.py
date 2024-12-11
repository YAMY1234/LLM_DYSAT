import re
import os
import itertools
from collections import defaultdict
from itertools import islice, chain

import networkx as nx
import numpy as np
import pickle as pkl
from scipy.sparse import csr_matrix

from datetime import datetime
from datetime import timedelta
import dateutil.parser


def lines_per_n(f, n):
    for line in f:
        yield ''.join(chain([line], itertools.islice(f, n - 1)))


def getDateTimeFromISO8601String(s):
    d = dateutil.parser.parse(s)
    return d


if __name__ == "__main__":
    # def UserItemNet_snapshot():
    data = 'movielens 100k'
    name1 = 'train'
    name2 = 'test'
    save_path = r'F:\models\DySAT-Transformer\DARec_Transformer\dataset\{}\sp_npz\graph.pkl'.format(data)
    # links_1 = np.loadtxt(r'F:\models\DySAT-Transformer\DySAT_Transformer\dataset\{}\{}.txt'.format(data, name1), dtype=int, delimiter=',', skiprows=0, usecols=(0, 1, 2), unpack=False, encoding='utf-8')
    # links_2 = np.loadtxt(r'F:\models\DySAT-Transformer\DySAT_Transformer\dataset\{}\{}.txt'.format(data, name2), dtype=int, delimiter=',', skiprows=0, usecols=(0, 1, 2), unpack=False, encoding='utf-8')
    links_3 = np.loadtxt(r'F:\models\DySAT-Transformer\DARec_Transformer\dataset\movielens 100k\u.data', dtype=int, delimiter='\t', skiprows=0, usecols=(0, 1, 3), unpack=False, encoding='utf-8')
    # links_3 = np.vstack([links_1, links_2])

    u_map = {j: i for i, j in enumerate(set(links_3[:, 0]))}
    p_map = {j: i for i, j in enumerate(set(links_3[:, 1]))}

    idx_u_map = np.array(list(map(u_map.get, links_3[:, 0])))
    idx_p_map = np.array(list(map(p_map.get, links_3[:, 1])))
    links_4 = np.dstack([idx_u_map, idx_p_map, links_3[:, 2]])[0]
    links = np.array(sorted(links_4, key=lambda x:x[2]))
    np.savetxt(r'F:\models\DySAT-Transformer\DARec_Transformer\dataset\{}\mv100k_time.txt'.format(data), links, fmt="%d", delimiter='\t')

    ts = links[:, 2]

    print(min(ts), max(ts))
    print("# interactions", links.shape[0])
    links = sorted(links,key=lambda x: x[2])
    links.sort(key=lambda x: x[2])

    # split edges
    SLICE_MONTHS = 1
    step_times = 20
    START_DATE = datetime.fromtimestamp(min(ts))  # + timedelta(200)
    END_DATE = datetime.fromtimestamp(max(ts))  # - timedelta(200)
    Date_DATE = (END_DATE-START_DATE).days//step_times
    print("Spliting Time Interval: \n Start Time : {}, End Time : {}".format(START_DATE, END_DATE))

    Date_list = [[] for date in range(Date_DATE+1)]

    for (a, b, time) in links:
        datetime_object = datetime.fromtimestamp(time)
        if datetime_object > END_DATE:
            months_diff = (END_DATE - START_DATE).days // step_times
        else:
            months_diff = (datetime_object - START_DATE).days // step_times
        Date_list[months_diff].append([a, b])

    all_edge = len(links)
    train_edge = sum([len(Date_list[i]) for i in range(len(Date_list)-2)])
    val_edge = len(Date_list[-2])
    # val_edge = sum([len(Date_list[i]) for i in range(len(Date_list)-2, len(Date_list)-2)])
    # test_edge = sum([len(Date_list[i]) for i in range(len(Date_list)-2, len(Date_list))])
    test_edge = len(Date_list[-1])



    for idx, Date_li in enumerate(Date_list):
        if (idx > 0) and (idx < (len(Date_list)-2)):
            Date_list[idx] # .extend(Date_list[idx-1]) 不交叉
        elif idx == len(Date_list)-2:
            Date_list[-1].extend(Date_list[idx][int(0.6*len(Date_list[idx])):])
            Date_list[-2] = Date_list[-2][:int(0.6*len(Date_list[idx]))]
        else:
            continue

    print('train_edge:{} %.3'.format(train_edge/all_edge),
          'val_edge:{} %'.format(len(Date_list[-2])/all_edge),
          'test_edge:{} %'.format(len(Date_list[-1])/all_edge))

    np.savetxt(r'F:\models\DySAT-Transformer\DARec_Transformer\dataset\movielens 100k\\' + 'train_.txt', np.array(Date_list[-3]), fmt="%d", delimiter=',')
    np.savetxt(r'F:\models\DySAT-Transformer\DARec_Transformer\dataset\movielens 100k\\' + 'test_.txt', np.array(Date_list[-1]), fmt="%d", delimiter=',')
    np.savetxt(r'F:\models\DySAT-Transformer\DARec_Transformer\dataset\movielens 100k\\' + 'val_.txt', np.array(Date_list[-2]), fmt="%d", delimiter=',')

    UserItemNet = []
    for date_graph in Date_list:
        UserItemNet.append(csr_matrix((np.ones(len(date_graph)), (np.array(date_graph)[:, 0], np.array(date_graph)[:, 1])),
                                 shape=(len(u_map), len(p_map)))) # 生成一个用户项交互矩阵R


    with open(save_path, "wb") as f:
        pkl.dump(UserItemNet, f)
        print("Processed Data Saved at {}".format(save_path))