import torch
import math
import pandas as pd
import numpy as np
from itertools import combinations, permutations
import networkx as nx
import matplotlib.pyplot as plt
from evaluation import *
from CI_test import newCI_test, oldCI_test, getCorr
import scipy.stats as st
import argparse
import time
import copy


def has_path(num_nodes, adj_, source, dest):
    queue = np.zeros([num_nodes]).astype('int')
    vis = np.zeros([num_nodes]).astype('bool')
    queue_l = 0
    queue_r = 1
    queue[0] = source
    vis[source] = True

    while (queue_l < queue_r):
        cur = queue[queue_l]
        queue_l += 1
        if cur == dest:
            return True
        for nxt in range(num_nodes):
            if (vis[nxt] == True):
                continue
            if not (adj_[cur][nxt] == 1 and adj_[nxt][cur] == 0):
                continue
            queue[queue_r] = nxt
            queue_r += 1
            vis[nxt] = True

    return False


def PC(data, CItest_type, max_size, alpha):
    num_nodes = len(data)
    # prepare a complete undirected graph
    adj = np.zeros([num_nodes, num_nodes]).astype('int')
    for (x, y) in permutations(range(num_nodes), 2):
        adj[x][y] = 1
    sep_set = [[tuple() for j in range(num_nodes)] for i in range(num_nodes)]

    # calc skeleton
    for size in range(max_size):
        for (x, y) in permutations(range(num_nodes), 2):
            if (adj[x][y] == 0):
                continue
            temp = []
            for z in range(num_nodes):
                if (adj[x][z] == 1 and z != y):
                    temp.append(z)
            temp = tuple(temp)
            for sep in combinations(temp, size):
                if CItest_type == 'new':
                    isCI = newCI_test(data, x, y, sep, alpha)
                else:
                    isCI = oldCI_test(data, max_size, x, y, sep, alpha)
                if isCI:
                    adj[x][y] = 0
                    adj[y][x] = 0
                    sep_set[x][y] += sep
                    break

    # identify V-structure
    for (x, z) in combinations(range(num_nodes), 2):
        if adj[x][z] != 1 or adj[z][x] != 1:
            continue
        for y in range(num_nodes):
            if x == y or z == y:
                continue
            if adj[x][y] != 0 or adj[y][x] != 0 or adj[y][z] != 1 or adj[z][y] != 1:
                continue
            if not z in sep_set[x][y]:
                adj[z][x] = adj[z][y] = 0

    # Transport the direction
    # while True:
    #     change_bool = False
    #
    #     for (x, y) in permutations(range(num_nodes), 2):
    #         if not (adj[x][y] == 1 and adj[y][x] == 0):
    #             continue
    #         for z in set(range(num_nodes)) - set([x, y]):
    #             if not (adj[y][z] == 1 and adj[z][y] == 1):
    #                 continue
    #             if not (adj[x][z] == adj[z][x] == 0):
    #                 continue
    #             adj[z][y] = 0
    #             change_bool = True
    #
    #     for (x, y) in permutations(range(num_nodes), 2):
    #         if not has_path(num_nodes, adj, x, y):
    #             continue
    #         if not (adj[x][y] == 1 and adj[y][x] == 1):
    #             continue
    #         adj[y][x] = 0
    #         change_bool = True
    #
    #     if change_bool == False:
    #         break

    return adj


def show_graph(data,num_nodes,adj_test):

    print("total nodes:", num_nodes)
    G_zero = nx.DiGraph()
    adj_zero = np.zeros([num_nodes, num_nodes])
    count_children = []
    count_father = []
    res = set()
    cf = set()
    for j in range(num_nodes):
        if adj_test[0][j] == 1:
            adj_zero[0][j]=1
            count_children.append(j)
            res.add(j)

        # if adj_test[0][j] == 1:
        #     for k in range(num_nodes):
        #         if adj_test[k][j] == 1 and k != 0:
        #             cf.add(k)
        #             adj_zero[k][j] = 1
        #             G_zero.add_edge(k,j)
        #             res.add(k)

        if adj_test[j][0]==1:
            adj_zero[j][0] = 1
            G_zero.add_edge(j,0)
            count_father.append(j)
            res.add(j)

    # adj_two means superset MB
    adj_two = np.matmul(adj_zero, adj_zero)
    two = set()
    H = len(adj_two)
    C = len(adj_two[0])
    for i in range(H):
        for j in range(C):
            if adj_two[i][j] != 0:
                two.add(i)
                two.add(j)

    result = res.union(two)
    result = list(result)
    if len(result) == 0:
        return 0
    if 0 in result:
        result.remove(0)
        print("total supersetMB number is {}, they are{}".format(len(result), result))
    else:
        print("total supersetMB number is {}, they are{}".format(len(result), result))
        result.append(0)
        result.sort()
        return result

# main
data =np.loadtxt(open("TCGA_LUAD_fs100.csv", "rb"), delimiter=",", skiprows=1)
CItest_type = 'new'
max_size = 1
alpha = 0.05
num_node = 101
data_type = data[:,0]
data -= np.mean(data, axis=0)
data = data.transpose()
adj_test = PC(data, CItest_type, max_size, alpha)
res = show_graph(data, num_node, adj_test)
data1 = pd.read_csv("TCGA_LUAD_fs100.csv",header=None, usecols=res)
print(data1.shape)
np.savetxt("test.csv", data1, delimiter=",")



