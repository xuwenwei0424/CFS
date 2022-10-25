from itertools import combinations, permutations

def getRecall_skeleton(num_nodes, adj_truth, adj_test):
    tot = 0
    cnt = 0
    for (i, j) in combinations(range(num_nodes), 2):
        if adj_truth[i][j] == 1 or adj_truth[j][i] == 1:
            tot += 1
            if adj_test[i][j] == 1 or adj_test[j][i] == 1:
                cnt += 1
    if tot != 0:
        return cnt * 1.0 / tot
    else:
        return -1


def getPrec_skeleton(num_nodes, adj_truth, adj_test):
    tot = 0
    cnt = 0
    for (i, j) in combinations(range(num_nodes), 2):
        if adj_test[i][j] == 1 or adj_test[j][i] == 1:
            tot += 1
            if adj_truth[i][j] == 1 or adj_truth[j][i] == 1:
                cnt += 1
    if tot != 0:
        return cnt * 1.0 / tot
    else:
        return -1


def getRecall_structrue(num_nodes, adj_truth, adj_test):
    tot = 0
    cnt = 0
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj_truth[i][j] == 1:
                tot += 1
                if adj_test[i][j] == 1:
                    cnt += 1
    if tot != 0:
        return cnt * 1.0 / tot
    else:
        return -1


def getPrec_structrue(num_nodes, adj_truth, adj_test):
    tot = 0
    cnt = 0
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj_test[i][j] == 1:
                tot += 1
                if adj_truth[i][j] == 1:
                    cnt += 1
    if tot != 0:
        return cnt * 1.0 / tot
    else:
        return -1
