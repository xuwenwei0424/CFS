import numpy as np
import math
import scipy.stats as st


def getCorr(data, x, y):
    mu_x = np.mean(data[x])
    mu_y = np.mean(data[y])
    num_nodes = len(data[x])

    sum_0 = 0
    sum_x = 0
    sum_y = 0

    for i in range(num_nodes):
        sum_0 += (data[x][i] - mu_x) * (data[y][i] - mu_y)
        sum_x += (data[x][i] - mu_x) * (data[x][i] - mu_x)
        sum_y += (data[y][i] - mu_y) * (data[y][i] - mu_y)

    return sum_0 / (math.sqrt(sum_x) * math.sqrt(sum_y))


# https://en.wikipedia.org/wiki/Partial_correlation
# By using dynamic programming
'''
Input:
    x : int
    y : int 
    Z : tuple
    data : data matrix( dim: num_nodes * num_samples )
    max_size : the max size of the conditional set

Output:
    isCI : Bool (True: x is independent with y given Z)

Parameter: 
    alpha : significance level (default 0.05/0.01)
'''


def oldCI_test(data, max_size, x, y, Z, alpha):
    num_nodes = len(data)
    corr = np.zeros([num_nodes, num_nodes, max_size])
    vis = np.zeros([num_nodes, num_nodes, max_size], dtype=bool)
    if len(Z) == 0:
        val = getCorr(data, x, y)
        if math.fabs(val) < alpha:
            return True
        else:
            return False

    def getCorr_cond(x, y, z, k):  # k表示当前即将处理到z集合中下标为k的数据

        if (vis[x][y][k] == True):
            return corr[x][y][k]
        if (k == len(Z)):
            return getCorr(data, x, y)
        vis[x][y][k] = True
        val_1 = getCorr_cond(x, Z[k], z, k + 1)
        val_2 = getCorr_cond(Z[k], y, Z, k + 1)
        val = getCorr_cond(x, y, Z, k + 1)
        corr[x][y][k] = (val - val_1 * val_2) / (math.sqrt(1 - val_1 * val_1) * math.sqrt(1 - val_2 * val_2))
        # print('({},{},{}) {:.3f}'.format(i,j,k,corr[x][y][k]))
        return corr[x][y][k]

    val = getCorr_cond(x, y, Z, 0)
    if val < alpha:
        return True
    else:
        return False


# https://en.wikipedia.org/wiki/Partial_correlation
# By using hypothesis test
'''
Input:
    x : int
    y : int 
    Z : tuple
    data : data matrix( dim: num_nodes * num_samples )

Output:
    isCI : Bool (True: x is independent with y given Z)

Parameter: 
    alpha : significance level (default 0.05/0.01)
'''


def newCI_test(data, x, y, Z, alpha):
    data_x = np.transpose(data[x])  # num_samples,
    data_y = np.transpose(data[y])  # num_samples,
    data_Z = np.transpose(data[Z, :])  # num_samples * |Z|

    Z_nodes = len(Z)  # length of Z
    if Z_nodes == 0:
        val = getCorr(data, x, y)
        if math.fabs(val) < alpha:
            return True
        else:
            return False

    num_samples = len(data_Z)  # number of data samples
    arr_one = (np.ones([num_samples]))
    data_Z = np.insert(data_Z, 0, arr_one, axis=1)  # insert an all-ones column in the left

    wx = np.linalg.lstsq(data_Z, data_x, rcond=None)[
        0]  # wx is the answer of data_Z * X = data_x by using least square method
    wy = np.linalg.lstsq(data_Z, data_y, rcond=None)[
        0]  # wy is the answer of data_Z * X = data_y by using least square method

    rx = data_x - data_Z @ wx  # calc residual error of data_x
    ry = data_y - data_Z @ wy  # calc residual error of data_y

    pcc = num_samples * (np.transpose(rx) @ ry) - np.sum(rx) * np.sum(ry)
    pcc /= math.sqrt(num_samples * (np.transpose(rx) @ rx) - np.sum(rx) * np.sum(rx))
    pcc /= math.sqrt(num_samples * (np.transpose(ry) @ ry) - np.sum(ry) * np.sum(ry))

    zpcc = 0.5 * math.log((1 + pcc) / (1 - pcc))
    A = math.sqrt(num_samples - Z_nodes - 3) * math.fabs(zpcc)
    B = st.norm.ppf(
        1 - alpha / 2)  # Inverse Cumulative Distribution Function of normal Gaussian (parameter : 1-alpha/2)

    if A > B:
        return False
    else:
        return True