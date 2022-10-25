import numpy as np
import math


def getFunc(isLinear):
    def pow1(x):
        return (np.random.random() + 0.2) * x

    def pow2(x):
        return (np.random.random() + 0.2) * x * x

    def pow3(x):
        return x * x * x

    if isLinear == True:
        return pow1

    x = np.random.random()
    if x < 0.2:
        return math.sin
    elif x < 0.4:
        return math.cos
    elif x < 0.6:
        return math.tanh
    elif x < 0.8:
        return pow2
    else:
        return pow1


def generate(num_nodes, num_samples, isLinear):
    # init
    adj = np.zeros([num_nodes, num_nodes])
    in_edges = np.zeros([num_nodes])
    parents = []
    for i in range(num_nodes):
        parents.append([])
    child = []
    for i in range(num_nodes):
        child.append([])

    # generate DAG
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):

            x = np.random.random()
            if (x > 0.5):
                adj[i][j] = 1
                parents[j].append(i)
                child[i].append(j)
                in_edges[j] += 1

    # generate samples
    values = np.zeros([num_nodes, num_samples])
    queue = np.zeros([num_nodes]).astype('int')
    queue_l = 0
    queue_r = 0
    func_g = []
    func_f = []

    for i in range(num_nodes):

        if in_edges[i] == 0:
            temp = np.zeros([num_samples])
            for j in range(num_samples):
                temp[j] = np.random.normal(0, 1)
            values[i] = temp
            queue[queue_r] = i
            queue_r += 1
        func_g.append(getFunc(isLinear))
        func_f.append(getFunc(isLinear))

    # topological sort
    while queue_l < queue_r:
        node = queue[queue_l]
        queue_l += 1
        if len(parents[node]) != 0:
            for i in parents[node]:
                for j in range(num_samples):
                    values[node][j] += func_f[i](func_g[i](values[i][j]))

            for j in range(num_samples):
                values[node][j] = func_g[node](values[node][j])
                values[node][j] = func_f[node](values[node][j])
                values[node][j] += (np.random.random() - 0.5) * 0.4
        for j in child[node]:
            in_edges[j] -= 1
            if in_edges[j] == 0:
                queue[queue_r] = j
                queue_r += 1

    return values, adj
