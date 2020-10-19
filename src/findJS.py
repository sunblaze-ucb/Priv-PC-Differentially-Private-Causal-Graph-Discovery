# -*- coding: utf-8 -*-

import numpy as np
from indep_test import bincondKendall, discondKendall
from gsq.ci_tests import ci_test_bin, ci_test_dis
from scipy.stats import chi2

import queue

def g_square_bin(dm, x, y, s):
    """G square test for a binary data.

    Args:
        dm: the data matrix to be used (as a numpy.ndarray).
        x: the first node (as an integer).
        y: the second node (as an integer).
        s: the set of neibouring nodes of x and y (as a set()).

    Returns:
        p_val: the p-value of conditional independence.
    """

    def _calculate_tlog(x, y, s, dof, dm):
        nijk = np.zeros((2, 2, dof))
        s_size = len(s)
        z = []
        for z_index in range(s_size):
            z.append(s.pop())
            pass
        for row_index in range(0, dm.shape[0]):
            i = dm[row_index, x]
            j = dm[row_index, y]
            k = []
            k_index = 0
            for z_index in range(s_size):
                k_index += dm[row_index, z[z_index]] * int(pow(2, z_index))
                pass
            nijk[i, j, k_index] += 1
            pass
        nik = np.ndarray((2, dof))
        njk = np.ndarray((2, dof))
        for k_index in range(dof):
            nik[:, k_index] = nijk[:, :, k_index].sum(axis = 1)
            njk[:, k_index] = nijk[:, :, k_index].sum(axis = 0)
            pass
        nk = njk.sum(axis = 0)
        tlog = np.zeros((2, 2 , dof))
        tlog.fill(np.nan)
        for k in range(dof):
            tx = np.array([nik[:,k]]).T
            ty = np.array([njk[:,k]])
            tdijk = tx.dot(ty)
            tlog[:,:,k] = nijk[:,:,k] * nk[k] / tdijk
            pass
        return (nijk, tlog)

    row_size = dm.shape[0]
    s_size = len(s)
    dof = int(pow(2, s_size))
    row_size_required = 10 * dof
    if row_size < row_size_required:
        return 1
    nijk = None
    if s_size < 6:
        if s_size == 0:
            nijk = np.zeros((2, 2))
            for row_index in range(0, dm.shape[0]):
                i = dm[row_index, x]
                j = dm[row_index, y]
                nijk[i, j] += 1
                pass
            tx = np.array([nijk.sum(axis = 1)]).T
            ty = np.array([nijk.sum(axis = 0)])
            tdij = tx.dot(ty)
            tlog = nijk * row_size / tdij
            pass
        if s_size > 0:
            nijk, tlog = _calculate_tlog(x, y, s, dof, dm)
            pass
        pass
    else:
        nijk = np.zeros((2, 2, 1))
        i = dm[0, x]
        j = dm[0, y]
        k = []
        for z in s:
            k.append(dm[:,z])
            pass
        k = np.array(k).T
        parents_count = 1
        parents_val = np.array([k[0,:]])
        nijk[i, j, parents_count - 1] = 1
        for it_sample in range(1, row_size):
            is_new = True
            i = dm[it_sample, x]
            j = dm[it_sample, y]
            tcomp = parents_val[:parents_count,:] == k[it_sample,:]
            for it_parents in range(parents_count):
                if np.all(tcomp[it_parents,:]):
                    nijk[i, j, it_parents] += 1
                    is_new = False
                    break
                pass
            if is_new is True:
                parents_count += 1
                parents_val = np.r_[parents_val, [k[it_sample,:]]]
                nnijk = np.zeros((2,2,parents_count))
                for p in range(parents_count - 1):
                    nnijk[:,:,p] = nijk[:,:,p]
                nnijk[i, j, parents_count - 1] = 1
                nijk = nnijk
                pass
            pass
        nik = np.ndarray((2, parents_count))
        njk = np.ndarray((2, parents_count))
        for k_index in range(parents_count):
            nik[:, k_index] = nijk[:, :, k_index].sum(axis = 1)
            njk[:, k_index] = nijk[:, :, k_index].sum(axis = 0)
            pass
        nk = njk.sum(axis = 0)
        tlog = np.zeros((2, 2 , parents_count))
        tlog.fill(np.nan)
        for k in range(parents_count):
            tX = np.array([nik[:,k]]).T
            tY = np.array([njk[:,k]])
            tdijk = tX.dot(tY)
            tlog[:,:,k] = nijk[:,:,k] * nk[k] / tdijk
            pass
        pass
    log_tlog = np.log(tlog)
    G2 = np.nansum(2 * nijk * log_tlog)
    p_val = chi2.sf(G2, dof)
    if s_size == 0:
        nijk = nijk.reshape((nijk.shape[0], nijk.shape[1], 1))
        log_tlog = log_tlog.reshape((log_tlog.shape[0], log_tlog.shape[1], 1))
    
    return G2, p_val, nijk, log_tlog

def g_square_dis(dm, x, y, s):
    """G square test for discrete data.

    Args:
        dm: the data matrix to be used (as a numpy.ndarray).
        x: the first node (as an integer).
        y: the second node (as an integer).
        s: the set of neibouring nodes of x and y (as a set()).
        levels: levels of each column in the data matrix
            (as a list()).

    Returns:
        p_val: the p-value of conditional independence.
    """
    levels = np.amax(dm, axis=0) + 1
    def _calculate_tlog(x, y, s, dof, levels, dm):
        prod_levels = np.prod(list(map(lambda x: levels[x], s)))
        nijk = np.zeros((levels[x], levels[y], prod_levels))
        s_size = len(s)
        z = []
        for z_index in range(s_size):
            z.append(s.pop())
            pass
        for row_index in range(dm.shape[0]):
            i = dm[row_index, x]
            j = dm[row_index, y]
            k = []
            k_index = 0
            for s_index in range(s_size):
                if s_index == 0:
                    k_index += dm[row_index, z[s_index]]
                else:
                    lprod = np.prod(list(map(lambda x: levels[x], z[:s_index])))
                    k_index += (dm[row_index, z[s_index]] * lprod)
                    pass
                pass
            nijk[i, j, k_index] += 1
            pass
        nik = np.ndarray((levels[x], prod_levels))
        njk = np.ndarray((levels[y], prod_levels))
        for k_index in range(prod_levels):
            nik[:, k_index] = nijk[:, :, k_index].sum(axis = 1)
            njk[:, k_index] = nijk[:, :, k_index].sum(axis = 0)
            pass
        nk = njk.sum(axis = 0)
        tlog = np.zeros((levels[x], levels[y], prod_levels))
        tlog.fill(np.nan)
        for k in range(prod_levels):
            tx = np.array([nik[:, k]]).T
            ty = np.array([njk[:, k]])
            tdijk = tx.dot(ty)
            tlog[:, :, k] = nijk[:, :, k] * nk[k] / tdijk
            pass
        return (nijk, tlog)

    row_size = dm.shape[0]
    s_size = len(s)
    dof = ((levels[x] - 1) * (levels[y] - 1)
           * np.prod(list(map(lambda x: levels[x], s))))
    row_size_required = 10 * dof
    nijk = None
    if s_size < 5:
        if s_size == 0:
            nijk = np.zeros((levels[x], levels[y]))
            for row_index in range(row_size):
                i = dm[row_index, x]
                j = dm[row_index, y]
                nijk[i, j] += 1
                pass
            tx = np.array([nijk.sum(axis = 1)]).T
            ty = np.array([nijk.sum(axis = 0)])
            tdij = tx.dot(ty)
            tlog = nijk * row_size / tdij
            pass
        if s_size > 0:
            nijk, tlog = _calculate_tlog(x, y, s, dof, levels, dm)
            pass
        pass
    else:
        nijk = np.zeros((levels[x], levels[y], 1))
        i = dm[0, x]
        j = dm[0, y]
        k = []
        for z in s:
            k.append(dm[:, z])
            pass
        k = np.array(k).T
        parents_count = 1
        parents_val = np.array([k[0, :]])
        nijk[i, j, parents_count - 1] = 1
        for it_sample in range(1, row_size):
            is_new = True
            i = dm[it_sample, x]
            j = dm[it_sample, y]
            tcomp = parents_val[:parents_count, :] == k[it_sample, :]
            for it_parents in range(parents_count):
                if np.all(tcomp[it_parents, :]):
                    nijk[i, j, it_parents] += 1
                    is_new = False
                    break
                pass
            if is_new is True:
                parents_count += 1
                parents_val = np.r_[parents_val, [k[it_sample, :]]]
                nnijk = np.zeros((levels[x], levels[y], parents_count))
                for p in range(parents_count - 1):
                    nnijk[:, :, p] = nijk[:, :, p]
                    pass
                nnijk[i, j, parents_count - 1] = 1
                nijk = nnijk
                pass
            pass
        nik = np.ndarray((levels[x], parents_count))
        njk = np.ndarray((levels[y], parents_count))
        for k_index in range(parents_count):
            nik[:, k_index] = nijk[:, :, k_index].sum(axis = 1)
            njk[:, k_index] = nijk[:, :, k_index].sum(axis = 0)
            pass
        nk = njk.sum(axis = 0)
        tlog = np.zeros((levels[x], levels[y], parents_count))
        tlog.fill(np.nan)
        for k in range(parents_count):
            tx = np.array([nik[:, k]]).T
            ty = np.array([njk[:, k]])
            tdijk = tx.dot(ty)
            tlog[:, :, k] = nijk[:, :, k] * nk[k] / tdijk
            pass
        pass
    log_tlog = np.log(tlog)
    G2 = np.nansum(2 * nijk * log_tlog)
    if dof == 0:
        p_val = 1
    else:
        p_val = chi2.sf(G2, dof)

    if s_size == 0:
        nijk = nijk.reshape((nijk.shape[0], nijk.shape[1], 1))
        log_tlog = log_tlog.reshape((log_tlog.shape[0], log_tlog.shape[1], 1))
    return G2, p_val, nijk, log_tlog


def findJs(x, y, S, alpha, dm, task='bin', verbose = False):
    if task == 'bin':
        indep_test_func = g_square_bin
        dof = int(pow(2, len(S)))
    else:
        indep_test_func = g_square_dis
        levels = np.amax(dm, axis=0) + 1
        dof = ((levels[x] - 1) * (levels[y] - 1)
            * np.prod(list(map(lambda x: levels[x], S))))

    G2, pval, nijk, log_tlog = indep_test_func(dm, x, y, S)
    n = dm.shape[0]
    if pval == alpha:
        return 0
    if pval > alpha:
        positive = 1
    else:
        positive = -1
    nik = np.sum(nijk, axis=1)
    njk = np.sum(nijk, axis=0)
    nk = np.sum(njk, axis=0)
    dims = nijk.shape

    threshold = chi2.isf(alpha, dof)
    temp_nij = np.zeros(dims[0:2])
    temp_log = np.zeros(dims[0:2])
    direction = np.zeros(dims)

    step_nijk = nijk.copy()
    step_nik = nik.copy()
    step_njk = njk.copy()
    step_nk = nk.copy()
    step_G2 = 2 * nijk * log_tlog # 2x2xk
    step_G2 = np.nansum(np.nansum(step_G2, axis=0),axis=0)
    steps = 0
    change = False
    
    while(change == False):
        for k in range(dims[2]):
            temp_n = step_nk[k] - 1
            for i in range(dims[0]):
                temp_ni = step_nik[:, k].copy()
                temp_ni[i] = temp_ni[i] - 1
                for j in range(dims[1]):
                    temp_nj = step_njk[:, k].copy()
                    temp_nj[j] = temp_nj[j] - 1
                    temp_nij = step_nijk[:, :, k].copy()

                    if (temp_nij[i, j] == 0):
                        direction[i, j, k] = -np.Inf
                    else:
                        temp_nij[i, j] = temp_nij[i, j] - 1
                        temp_ni = temp_ni.reshape((-1, 1))
                        temp_nj = temp_nj.reshape((-1, 1)).T
                        temp_log = temp_n * (temp_nij / np.dot(temp_ni, temp_nj))
                        temp_G2 = 2 * temp_nij * np.log(temp_log)
                        direction[i,j,k] = positive * (np.nansum(temp_G2) - step_G2[k])

        choose_direction = np.unravel_index(direction.argmax(), direction.shape)
        if (positive * (np.sum(step_G2) + positive * direction[choose_direction]) >= positive * threshold):
            JSdist = positive * steps
            change = True
        elif (steps == n-1):
            JSdist = positive * steps
            change = True
        else:
            steps += 1
            step_nijk[choose_direction] = step_nijk[choose_direction] - 1
            step_G2[choose_direction[-1]] = step_G2[choose_direction[-1]] + positive * direction[choose_direction]
            step_nik = np.sum(step_nijk, axis=1)
            step_njk = np.sum(step_nijk, axis=0)
            step_nk = np.sum(step_njk, axis=0)
    return JSdist

def NumSig(JSscore, pval, alpha):

    Z = zip(JSscore, pval)
    Z = sorted(Z, reverse=True)

    JSscore_new , pval_new = zip(*Z)
    qscore = np.zeros(len(JSscore) + 1)
    if pval_new[0] >= alpha:
        qscore[0] = -JSscore_new[0] - 1
    else:
        qscore[0] = -JSscore_new[0]

    if pval_new[-1] < alpha:
        qscore[-1] = JSscore_new[-1] - 1
    else:
        qscore[-1] = JSscore_new[-1]

    for i in range(1, len(JSscore_new)):
        if pval_new[i-1] > alpha and pval_new[i] <= alpha:
            qscore[i] = min(JSscore_new[i-1], -JSscore_new[i]) - 1
        elif pval_new[i-1] > alpha and pval_new[i] > alpha:
            qscore[i] = -JSscore_new[i]
        else:
            qscore[i] = JSscore_new[i-1]

    return qscore
