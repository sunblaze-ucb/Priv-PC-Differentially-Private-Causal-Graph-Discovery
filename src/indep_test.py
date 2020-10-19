# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats import norm
from Kendalltaua import kendalltaua
from gsq.ci_tests import ci_test_bin, ci_test_dis

def wrapped_ci_test_bin(dm, i, j, k, **kwargs):
    p_val = ci_test_bin(dm, i, j, k, **kwargs)
    return None, p_val

def wrapped_ci_test_dis(dm, i, j, k, **kwargs):
    p_val = ci_test_dis(dm, i, j, k, **kwargs)
    return None, p_val

def bincondKendall(data_matrix, x, y, k, **kwargs):
    s_size = len(k)
    row_size = data_matrix.shape[0]
    if s_size == 0:
        (tau, pval), _, _ = kendalltaua(data_matrix[:,x], data_matrix[:,y])
        tau = tau * np.sqrt(9.0 * row_size * (row_size - 1) / (4*row_size+10))
        pval = norm.sf(np.abs(tau))
        return tau, pval
    z = []
    for z_index in range(s_size):
        z.append(k.pop())
        pass

    dm_unique = np.unique(data_matrix[:, z], axis=0)
    sumwk = 0
    sumweight = 0
    tau = 0
    pval = 0
    for split_k in dm_unique:
        index = np.ones((row_size),dtype=bool)
        for i in range(s_size):
            index = ((data_matrix[..., z[i]] == split_k[i]) & index)

        new_dm = data_matrix[index, :]
        nk = new_dm.shape[0]
        if nk <= 2:
            continue
        (condtau, condpval), cntx, cnty = kendalltaua(new_dm[:, x], new_dm[:, y])
        if np.isnan(condpval):
            continue
        sigma0_sq = (4.0 * nk + 10) / (9.0 * nk * (nk-1.0))
        tau += condtau / sigma0_sq
        sumwk += 1.0 / sigma0_sq

    tau /= np.sqrt(sumwk)
    pval = norm.sf(np.abs(tau))

    return tau, pval

def discondKendall(data_matrix, x, y, k, **kwargs):
    s_size = len(k)
    row_size = data_matrix.shape[0]
    if s_size == 0:
        (tau, pval), _, _ = kendalltaua(data_matrix[:,x], data_matrix[:,y])
        tau = tau * np.sqrt(9.0 * row_size * (row_size - 1) / (4*row_size+10))
        pval = norm.sf(np.abs(tau))
        return tau, pval
    z = []
    for z_index in range(s_size):
        z.append(k.pop())
        pass

    dm_unique = np.unique(data_matrix[:, z], axis=0)
    sumwk = 0
    sumweight = 0
    tau = 0
    pval = 0
    for split_k in dm_unique:
        index = np.ones((row_size),dtype=bool)
        for i in range(s_size):
            index = ((data_matrix[..., z[i]] == split_k[i]) & index)

        new_dm = data_matrix[index, :]
        nk = new_dm.shape[0]
        if nk <= 2:
            continue
        (condtau, condpval), cntx, cnty = kendalltaua(new_dm[:, x], new_dm[:, y])
        if np.isnan(condpval):
            continue
        sigma0_sq = (4.0 * nk + 10) / (9.0 * nk * (nk-1.0))
        tau += condtau / sigma0_sq
        sumwk += 1.0 / sigma0_sq

    tau /= np.sqrt(sumwk)
    pval = norm.sf(np.abs(tau))

    return tau, pval
