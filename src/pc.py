# -*- coding: utf-8 -*-

from __future__ import print_function

from itertools import combinations, permutations
import logging

import time
import math
import numpy as np
import networkx as nx
from scipy.integrate import quad
from indep_test import bincondKendall, discondKendall, wrapped_ci_test_bin, wrapped_ci_test_dis
from findJS import findJs, NumSig
from scipy import stats, integrate
from scipy.special import comb
from scipy.optimize import minimize
from dataset import *
from gsq.ci_tests import ci_test_bin, ci_test_dis
from gsq.gsq_testdata import bin_data, dis_data
import argparse

_logger = logging.getLogger(__name__)

def _create_complete_graph(node_ids):
    """Create a complete graph from the list of node ids.
    Args:
        node_ids: a list of node ids
    Returns:
        An undirected graph (as a networkx.Graph)
    """
    g = nx.Graph()
    g.add_nodes_from(node_ids)
    for (i, j) in combinations(node_ids, 2):
        g.add_edge(i, j)
    return g

def estimate_skeleton(indep_test_func, data_matrix, alpha, **kwargs):
    """Estimate a skeleton graph from the statistis information.
    Args:
        indep_test_func: the function name for a conditional
            independency test.
        data_matrix: data (as a numpy array).
        alpha: the significance level.
        kwargs:
            'max_reach': maximum value of l (see the code).  The
                value depends on the underlying distribution.
            'method': if 'stable' given, use stable-PC algorithm
                (see [Colombo2014]).
            'init_graph': initial structure of skeleton graph
                (as a networkx.Graph). If not specified,
                a complete graph is used.
            other parameters may be passed depending on the
                indep_test_func()s.
    Returns:
        g: a skeleton graph (as a networkx.Graph).
        sep_set: a separation set (as an 2D-array of set()).
    [Colombo2014] Diego Colombo and Marloes H Maathuis. Order-independent
    constraint-based causal structure learning. In The Journal of Machine
    Learning Research, Vol. 15, pp. 3741-3782, 2014.
    """

    def method_stable(kwargs):
        return ('method' in kwargs) and kwargs['method'] == "stable"

    node_ids = range(data_matrix.shape[1])
    node_size = data_matrix.shape[1]
    sep_set = [[set() for i in range(node_size)] for j in range(node_size)]
    if 'init_graph' in kwargs:
        g = kwargs['init_graph']
        if not isinstance(g, nx.Graph):
            raise ValueError
        elif not g.number_of_nodes() == len(node_ids):
            raise ValueError('init_graph not matching data_matrix shape')
        for (i, j) in combinations(node_ids, 2):
            if not g.has_edge(i, j):
                sep_set[i][j] = None
                sep_set[j][i] = None
    else:
        g = _create_complete_graph(node_ids)

    l = 0
    q = 1.0 / 10.0
    while True:
        cont = False
        remove_edges = []
        for (i, j) in permutations(node_ids, 2):
            adj_i = list(g.neighbors(i))
            if j not in adj_i:
                continue
            else:
                adj_i.remove(j)
            if len(adj_i) >= l:
                _logger.debug('testing %s and %s' % (i,j))
                _logger.debug('neighbors of %s are %s' % (i, str(adj_i)))
            if len(adj_i) < l:
                continue
            for k in combinations(adj_i, l):
                _logger.debug('indep prob of %s and %s with subset %s'
                                % (i, j, str(k)))
                tau, p_val = indep_test_func(data_matrix, i, j, set(k), **kwargs)
                _logger.debug('tau is %s' % str(tau))
                if p_val > alpha:
                    if g.has_edge(i, j):
                        _logger.debug('p: remove edge (%s, %s)' % (i, j))
                        if method_stable(kwargs):
                            remove_edges.append((i, j))
                        else:
                            g.remove_edge(i, j)
                    sep_set[i][j] |= set(k)
                    sep_set[j][i] |= set(k)
                    break
            cont = True
        l += 1
        if method_stable(kwargs):
            g.remove_edges_from(remove_edges)
        if cont is False:
            break
        if ('max_reach' in kwargs) and (l > kwargs['max_reach']):
            break

    return (g, sep_set)

def estimate_skeleton_EM(indep_test_func, data_matrix, alpha, eps=1, delta=1e-3, task='bin', **kwargs):

    def method_stable(kwargs):
        return ('method' in kwargs) and kwargs['method'] == "stable"

    test_count = 0

    node_ids = range(data_matrix.shape[1])
    node_size = data_matrix.shape[1]
    sep_set = [[set() for i in range(node_size)] for j in range(node_size)]

    if 'init_graph' in kwargs:
        g = kwargs['init_graph']
        if not isinstance(g, nx.Graph):
            raise ValueError
        elif not g.number_of_nodes() == len(node_ids):
            raise ValueError('init_graph not matching data_matrix shape')
        for (i, j) in combinations(node_ids, 2):
            if not g.has_edge(i, j):
                sep_set[i][j] = None
                sep_set[j][i] = None
    else:
        g = _create_complete_graph(node_ids)

    l = 0
    count = 0
    while True:
        cont = False
        remove_edges = []
        for i in node_ids:
            adj_i = list(g.neighbors(i))
            if len(adj_i) < l + 1:
                continue
            count = count + 1
            ind_set, k_set, temp_count = findPrivInd(i, adj_i, l, data_matrix, eps/2, eps/2, indep_test_func, alpha, task, **kwargs)
            test_count += temp_count
            for j in range(len(ind_set)):
                if g.has_edge(i, ind_set[j]):
                    _logger.debug('p: remove edge (%s, %s)' % (i, j))
                    if method_stable(kwargs):
                        remove_edges.append((i, ind_set[j]))
                    else:
                        g.remove_edge(i, ind_set[j])
                sep_set[i][j] |= set(k_set[j])
                sep_set[j][i] |= set(k_set[j])
            cont = True
        l += 1
        if method_stable(kwargs):
            g.remove_edges_from(remove_edges)
        if cont is False:
            break
        if ('max_reach' in kwargs) and (l > kwargs['max_reach']):
            break

    # advanced composition
    eps_prime = np.sqrt(2*count*np.log(1/delta))*eps + count*eps*(np.exp(eps)-1)

    return (g, sep_set, eps_prime, delta, test_count)


def findPrivInd(i, adj_i, l, data_matrix, epsilon1, epsilon2, indep_test_func, alpha, task, **kwargs):
    
    test_count = 0
    
    q1 = dict()
    q2 = dict()
    pval = dict()
    K_set = dict()
    for j in adj_i:
        R1 = adj_i.copy()
        R1.remove(j)
        max_pval = -1
        for k in combinations(R1, l):
            _, temp_pval = indep_test_func(data_matrix, i, j, set(k), **kwargs)
            test_count += 1
            if temp_pval > max_pval:
                max_pval = temp_pval
                max_k = k
        q1[j] = findJs(i, j, set(max_k), alpha, data_matrix, task)
        pval[j] = max_pval
        K_set[j] = max_k
    R2 = range(len(adj_i))

    q2 = NumSig(q1.values(), pval.values(), alpha)

    beta = EM(q2, epsilon2, 1, data_matrix, R2)
    Ind_set = []
    re_K = []
    for t in range(beta):
        Vr = EM(q1, epsilon1 / beta, 1, data_matrix, adj_i)
        Ind_set.append(Vr)
        re_K.append(K_set[Vr])
        q1[Vr] = -10000

    return Ind_set, re_K, test_count


def EM(q, epsilon, S, D, R):

    prob = dict()
    for i in range(len(R)):
        r = R[i]
        prob[r] = epsilon * q[r] / 2.0 / S
    prob = np.array(list(prob.values()))
    if np.any(np.isinf(prob)):
        return R[np.where(prob>0)[0][0]]
    prob = prob - np.max(prob) + 10
    prob = np.exp(prob)
    prob = prob / prob.sum()
    index = np.random.choice(R, p=prob.ravel())

    return index

def estimate_skeleton_SVT(indep_test_func, data_matrix, alpha, eps=1, delta=None, **kwargs):

    def method_stable(kwargs):
        return ('method' in kwargs) and kwargs['method'] == "stable"

    test_count = 0

    node_ids = range(data_matrix.shape[1])
    n = data_matrix.shape[0]
    node_size = data_matrix.shape[1]
    sep_set = [[set() for i in range(node_size)] for j in range(node_size)]
    if 'init_graph' in kwargs:
        g = kwargs['init_graph']
        if not isinstance(g, nx.Graph):
            raise ValueError
        elif not g.number_of_nodes() == len(node_ids):
            raise ValueError('init_graph not matching data_matrix shape')
        for (i, j) in combinations(node_ids, 2):
            if not g.has_edge(i, j):
                sep_set[i][j] = None
                sep_set[j][i] = None
    else:
        g = _create_complete_graph(node_ids)

    l = 0
    count = 0
    if delta is None:
        delta = 1e-4
    S, _ = quad(lambda x: np.exp(-x**2/2) / np.sqrt(2*np.pi), 0, 6 / np.sqrt(n))
    sigma1 = 2 * S / eps
    sigma2 = 4 * sigma1

    T0 = alpha + np.random.laplace(0, sigma1)
    while True:
        cont = False
        remove_edges = []
        for (i, j) in permutations(node_ids, 2):
            adj_i = list(g.neighbors(i))
            if j not in adj_i:
                continue
            else:
                adj_i.remove(j)
            if len(adj_i) >= l:
                _logger.debug('testing %s and %s' % (i,j))
                _logger.debug('neighbors of %s are %s' % (i, str(adj_i)))
            if len(adj_i) < l:
                continue

            for k in combinations(adj_i, l):
                _logger.debug('indep prob of %s and %s with subset %s'
                                % (i, j, str(k)))
                v = np.random.laplace(0, sigma2)
                p_val = indep_test_func(data_matrix, i, j, set(k), **kwargs)[1] + v
                test_count += 1
                _logger.debug('p_val is %s' % str(p_val))

                if p_val < T0:
                    continue
                if p_val >= T0:
                    count += 1
                    if g.has_edge(i, j):
                        _logger.debug('p: remove edge (%s, %s)' % (i, j))
                        if method_stable(kwargs):
                            remove_edges.append((i, j))
                        else:
                            g.remove_edge(i, j)
                    sep_set[i][j] |= set(k)
                    sep_set[j][i] |= set(k)
                    T0 = alpha + np.random.normal(0, sigma1)
                    break
            cont = True
        l += 1
        if method_stable(kwargs):
            g.remove_edges_from(remove_edges)
        if cont is False:
            break
        if ('max_reach' in kwargs) and (l > kwargs['max_reach']):
            break

    # advanced composition
    eps_prime = np.sqrt(2*count*np.log(1/delta))*eps + count*eps*(np.exp(eps)-1)
 
    return (g, sep_set, eps_prime, delta, test_count)

def estimate_skeleton_probe_examine(indep_test_func, data_matrix, alpha, eps=1, delta=1e-3, bias=0.02, **kwargs):

    def method_stable(kwargs):
        return ('method' in kwargs) and kwargs['method'] == "stable"
    
    test_count = 0
    
    node_ids = range(data_matrix.shape[1])
    n = data_matrix.shape[0]
    node_size = data_matrix.shape[1]
    sep_set = [[set() for i in range(node_size)] for j in range(node_size)]
    if 'init_graph' in kwargs:
        g = kwargs['init_graph']
        if not isinstance(g, nx.Graph):
            raise ValueError
        elif not g.number_of_nodes() == len(node_ids):
            raise ValueError('init_graph not matching data_matrix shape')
        for (i, j) in combinations(node_ids, 2):
            if not g.has_edge(i, j):
                sep_set[i][j] = None
                sep_set[j][i] = None
    else:
        g = _create_complete_graph(node_ids)

    l = 0
    count = 0
    budget_split = 1.0 / 2.0
    eps1 = eps * budget_split
    def noise_scale(x):
        return np.sqrt(x) / np.log(x * (np.exp(eps1)-1) + 1)

    q = max(min(1. / minimize(noise_scale, [0.5], tol=1e-2).x[0], 1), 1. / 20.)
    eps2 = eps - eps1
    S, _ = quad(lambda x: np.exp(-x**2/2) / np.sqrt(2*np.pi), 0, 6 / np.sqrt(n))
    sigma1 = 2.0 * S / np.sqrt(q) / np.log((np.exp(eps1)-1.)/q + 1)
    sigma2 = 2 * sigma1
    sigma3 = S / eps2
    # bias = 9 * sigma1

    T0 = alpha - bias + np.random.laplace(0, sigma1)
    row_rand = np.arange(n)
    np.random.shuffle(row_rand)
    dm_subsampled = data_matrix[row_rand[0:int(n*q)]]
    while True:
        cont = False
        remove_edges = []
        
        for (i, j) in permutations(node_ids, 2):
            adj_i = list(g.neighbors(i))
            if j not in adj_i:
                continue
            else:
                adj_i.remove(j)
            if len(adj_i) >= l:
                _logger.debug('testing %s and %s' % (i,j))
                _logger.debug('neighbors of %s are %s' % (i, str(adj_i)))
            if len(adj_i) < l:
                continue

            for k in combinations(adj_i, l):
                _logger.debug('indep prob of %s and %s with subset %s'
                                % (i, j, str(k)))
                v = np.random.laplace(0, sigma2)
                p_val = indep_test_func(dm_subsampled, i, j, set(k), **kwargs)[1] + v
                test_count += 1
                _logger.debug('p_val is %s' % str(p_val))

                if p_val < T0:
                    continue
                if p_val >= T0:
                    count += 1
                    T0 = alpha - bias + np.random.laplace(0, sigma1)
                    np.random.shuffle(row_rand)
                    dm_subsampled = data_matrix[row_rand[0:int(n*q)]]
                    v = np.random.laplace(0, sigma3)
                    p_val = indep_test_func(data_matrix, i, j, set(k), **kwargs)[1] + v
                    test_count += 1
                    if p_val >= alpha:
                        if g.has_edge(i, j):
                            _logger.debug('p: remove edge (%s, %s)' % (i, j))
                            if method_stable(kwargs):
                                remove_edges.append((i, j))
                            else:
                                g.remove_edge(i, j)
                        sep_set[i][j] |= set(k)
                        sep_set[j][i] |= set(k)
                        break
            cont = True
        l += 1
        if method_stable(kwargs):
            g.remove_edges_from(remove_edges)
        if cont is False:
            break
        if ('max_reach' in kwargs) and (l > kwargs['max_reach']):
            break

    eps_prime1 = np.sqrt(2*count*np.log(2/delta))*eps2 + count*eps2*(np.exp(eps2)-1)
    eps_prime2 = np.sqrt(2*count*np.log(2/delta))*eps1 + count*eps1*(np.exp(eps1)-1)
    eps_prime = eps_prime1 + eps_prime2

    return (g, sep_set, eps_prime, 1e-3, test_count)

def estimate_cpdag(skel_graph, sep_set):
    """Estimate a CPDAG from the skeleton graph and separation sets
    returned by the estimate_skeleton() function.
    Args:
        skel_graph: A skeleton graph (an undirected networkx.Graph).
        sep_set: An 2D-array of separation set.
            The contents look like something like below.
                sep_set[i][j] = set([k, l, m])
    Returns:
        An estimated DAG.
    """
    dag = skel_graph.to_directed()
    node_ids = skel_graph.nodes()
    for (i, j) in combinations(node_ids, 2):
        adj_i = set(dag.successors(i))
        if j in adj_i:
            continue
        adj_j = set(dag.successors(j))
        if i in adj_j:
            continue
        if sep_set[i][j] is None:
            continue
        common_k = adj_i & adj_j
        for k in common_k:
            if k not in sep_set[i][j]:
                if dag.has_edge(k, i):
                    _logger.debug('S: remove edge (%s, %s)' % (k, i))
                    dag.remove_edge(k, i)
                if dag.has_edge(k, j):
                    _logger.debug('S: remove edge (%s, %s)' % (k, j))
                    dag.remove_edge(k, j)

    def _has_both_edges(dag, i, j):
        return dag.has_edge(i, j) and dag.has_edge(j, i)

    def _has_any_edge(dag, i, j):
        return dag.has_edge(i, j) or dag.has_edge(j, i)

    def _has_one_edge(dag, i, j):
        return ((dag.has_edge(i, j) and (not dag.has_edge(j, i))) or
                (not dag.has_edge(i, j)) and dag.has_edge(j, i))

    def _has_no_edge(dag, i, j):
        return (not dag.has_edge(i, j)) and (not dag.has_edge(j, i))

    # For all the combination of nodes i and j, apply the following
    # rules.
    old_dag = dag.copy()
    while True:
        for (i, j) in combinations(node_ids, 2):
            # Rule 1: Orient i-j into i->j whenever there is an arrow k->i
            # such that k and j are nonadjacent.
            #
            # Check if i-j.
            if _has_both_edges(dag, i, j):
                # Look all the predecessors of i.
                for k in dag.predecessors(i):
                    # Skip if there is an arrow i->k.
                    if dag.has_edge(i, k):
                        continue
                    # Skip if k and j are adjacent.
                    if _has_any_edge(dag, k, j):
                        continue
                    # Make i-j into i->j
                    _logger.debug('R1: remove edge (%s, %s)' % (j, i))
                    dag.remove_edge(j, i)
                    break

            # Rule 2: Orient i-j into i->j whenever there is a chain
            # i->k->j.
            #
            # Check if i-j.
            if _has_both_edges(dag, i, j):
                # Find nodes k where k is i->k.
                succs_i = set()
                for k in dag.successors(i):
                    if not dag.has_edge(k, i):
                        succs_i.add(k)
                # Find nodes j where j is k->j.
                preds_j = set()
                for k in dag.predecessors(j):
                    if not dag.has_edge(j, k):
                        preds_j.add(k)
                # Check if there is any node k where i->k->j.
                if len(succs_i & preds_j) > 0:
                    # Make i-j into i->j
                    _logger.debug('R2: remove edge (%s, %s)' % (j, i))
                    dag.remove_edge(j, i)

            # Rule 3: Orient i-j into i->j whenever there are two chains
            # i-k->j and i-l->j such that k and l are nonadjacent.
            #
            # Check if i-j.
            if _has_both_edges(dag, i, j):
                # Find nodes k where i-k.
                adj_i = set()
                for k in dag.successors(i):
                    if dag.has_edge(k, i):
                        adj_i.add(k)
                # For all the pairs of nodes in adj_i,
                for (k, l) in combinations(adj_i, 2):
                    # Skip if k and l are adjacent.
                    if _has_any_edge(dag, k, l):
                        continue
                    # Skip if not k->j.
                    if dag.has_edge(j, k) or (not dag.has_edge(k, j)):
                        continue
                    # Skip if not l->j.
                    if dag.has_edge(j, l) or (not dag.has_edge(l, j)):
                        continue
                    # Make i-j into i->j.
                    _logger.debug('R3: remove edge (%s, %s)' % (j, i))
                    dag.remove_edge(j, i)
                    break

            # Rule 4: Orient i-j into i->j whenever there are two chains
            # i-k->l and k->l->j such that k and j are nonadjacent.
            #
            # However, this rule is not necessary when the PC-algorithm
            # is used to estimate a DAG.

        if nx.is_isomorphic(dag, old_dag):
            break
        old_dag = dag.copy()

    return dag

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    args = parser.parse_args()

    dm, g_answer = bn_data(args.dataset, size=100000)

    (g, sep_set) = estimate_skeleton(indep_test_func=wrapped_ci_test_dis,
                                     data_matrix=dm,
                                     alpha=0.01)
    g = estimate_cpdag(skel_graph=g, sep_set=sep_set)

    print('G-test edges are:', g.edges(), end='')
    if nx.is_isomorphic(g, g_answer):
        print(' => G-test GOOD')
    else:
        print(' => G-test WRONG')
        print('True edges should be:', g_answer.edges())

    (g_tau, sep_set) = estimate_skeleton(indep_test_func=discondKendall,
                                     data_matrix=dm,
                                     alpha=0.10)
    g_tau = estimate_cpdag(skel_graph=g_tau, sep_set=sep_set)

    print('Kendall tau edges are:', g_tau.edges(), end='')
    if nx.is_isomorphic(g_tau, g_answer):
        print(' => Kendall tau GOOD')
    else:
        print(' => Kendall tau WRONG')
        print('True edges should be:', g_answer.edges())
