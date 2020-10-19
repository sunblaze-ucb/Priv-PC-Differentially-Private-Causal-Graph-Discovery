from __future__ import print_function
from itertools import combinations, permutations
import logging
import time
import pickle
import argparse
import numpy as np
from tqdm import tqdm                        
from gsq.ci_tests import ci_test_bin, ci_test_dis
from gsq.gsq_testdata import bin_data, dis_data
import networkx as nx
from indep_test import bincondKendall, discondKendall
from pc import *
from dataset import *

_logger = logging.getLogger(__name__) 
pkl_template = '../results/pkl/%s_privpc.pkl'
ALPHA=0.10

def cal_precision(e1, e2):
    cnt = 0.0
    for e in e1:
        if e in e2:
            cnt = cnt + 1
    if cnt == 0:
        if len(e1) == 0:
            return 1
        return 0
    return cnt / len(e1)

def cal_recall(e1, e2):
    cnt = 0.0
    for e in e2:
        if e in e1:
            cnt = cnt + 1
    if cnt == 0:
        return 0
    return cnt / len(e2)

def cal_f1(e1, e2):
    precision = cal_precision(e1, e2)
    recall = cal_recall(e1, e2)
    numerator = 2*precision*recall
    if numerator == 0:
        return 0
    return numerator/(precision+recall)

def eval(dataset='earthquake', iters=10, epsilon=1, delta=1e-3, alpha=0.01, option='all'):

    run_time = {"priv-pc": [], "em": [], 'svt': []}
    f1 = {"priv-pc": [], "em": [], 'svt': []}
    privacy = {"priv-pc": [], "em": [], 'svt': []}
    indep_test_numbers = {"priv-pc": [], "em": [], 'svt': []}

    if dataset in ['asia', 'cancer', 'earthquake']:
        dm, g_answer = bn_data(dataset, size=100000)

        max_reach = max(min(np.int(np.log2(dm.shape[0]))-5, dm.shape[1]-2), 0)

        if option == 'all' or option == 'pae':
            print("Run Priv-PC algorithms for %d times..."%iters)
            for _ in tqdm(range(iters)):
                start = time.time()
                (g, sep_set, eps, _, test_number) = estimate_skeleton_probe_examine(indep_test_func=bincondKendall,
                                                             data_matrix=dm,
                                                             alpha=alpha,
                                                             eps=epsilon,
                                                             delta=delta,
                                                             max_reach=max_reach)
                g = estimate_cpdag(skel_graph=g, sep_set=sep_set)
                end = time.time()
                f1['priv-pc'].append(cal_f1(g.edges, g_answer.edges))
                run_time['priv-pc'].append(end-start)
                privacy['priv-pc'].append(eps)
                indep_test_numbers['priv-pc'].append(test_number)

        if option == 'all' or option == 'svt':
            print("Run SVT-PC algorithms for %d times..."%iters)
            for _ in tqdm(range(iters)):
                start = time.time()
                (g, sep_set, eps, _, test_number) = estimate_skeleton_SVT(indep_test_func=bincondKendall,
                                                             data_matrix=dm,
                                                             alpha=alpha,
                                                             eps=epsilon,
                                                             delta=delta,
                                                             max_reach=max_reach)
                g = estimate_cpdag(skel_graph=g, sep_set=sep_set)
                end = time.time()
                f1['svt'].append(cal_f1(g.edges, g_answer.edges))
                run_time['svt'].append(end-start)
                privacy['svt'].append(eps)
                indep_test_numbers['svt'].append(test_number)

        if option == 'all' or option == 'em':
            print("Run EM-PC algorithms for %d times..."%iters)
            for _ in tqdm(range(iters)):
                start = time.time()
                (g, sep_set, eps, _, test_number) = estimate_skeleton_EM(indep_test_func=wrapped_ci_test_bin,
                                                    data_matrix=dm,
                                                    alpha=alpha, 
                                                    eps=epsilon,
                                                    delta=delta,
                                                    task='bin')
                g = estimate_cpdag(skel_graph=g, sep_set=sep_set)
                end = time.time()
                f1['em'].append(cal_f1(g.edges, g_answer.edges))
                run_time['em'].append(end-start)
                privacy['em'].append(eps)
                indep_test_numbers['em'].append(test_number)

    elif dataset in ['survey', 'sachs', 'child', 'alarm']:

        dm, g_answer = bn_data(dataset, size=100000)
        max_reach = max(min(np.int(np.log2(dm.shape[0]))-5, dm.shape[1]-2), 0)
        # levels = np.amax(dm, axis=0) + 1
        # (g, sep_set) = estimate_skeleton(indep_test_func=wrapped_ci_test_dis,
        #                                     data_matrix=dm,
        #                                     alpha=alpha,
        #                                     levels=levels,
        #                                     max_reach=max_reach)
        # g = estimate_cpdag(skel_graph=g, sep_set=sep_set)
        # print(g.edges)
        if option == 'all' or option == 'pae':
            print("Run Priv-PC algorithms for %d times..."%iters)
            for _ in tqdm(range(iters)):
                start = time.time()
                (g, sep_set, eps, _, test_number) = estimate_skeleton_probe_examine(indep_test_func=discondKendall,
                                                            data_matrix=dm,
                                                            alpha=alpha,
                                                            eps=epsilon,
                                                            delta=delta,
                                                            max_reach=max_reach)
                g = estimate_cpdag(skel_graph=g, sep_set=sep_set)
                end = time.time()
                run_time['priv-pc'].append(end-start)
                f1['priv-pc'].append(cal_f1(g.edges, g_answer.edges))
                privacy['priv-pc'].append(eps)
                indep_test_numbers['priv-pc'].append(test_number)

        if option == 'all' or option == 'svt':
            print("Run SVT-PC algorithms for %d times..."%iters)
            for _ in tqdm(range(iters)):
                start = time.time()
                (g, sep_set, eps, _, test_number) = estimate_skeleton_SVT(indep_test_func=discondKendall,
                                                            data_matrix=dm,
                                                            alpha=alpha,
                                                            eps=epsilon,
                                                            delta=delta,
                                                            max_reach=max_reach)
                g = estimate_cpdag(skel_graph=g, sep_set=sep_set)
                end = time.time()
                run_time['svt'].append(end-start)
                f1['svt'].append(cal_f1(g.edges, g_answer.edges))
                privacy['svt'].append(eps)
                indep_test_numbers['svt'].append(test_number)

        if option == 'em' or option == 'all':
            print("Run EM-PC algorithms for %d times..."%iters)
            for _ in tqdm(range(iters)):
                start = time.time()
                (g, sep_set, eps, _, test_number) = estimate_skeleton_EM(indep_test_func=wrapped_ci_test_dis,
                                                    data_matrix=dm,
                                                    alpha=alpha, 
                                                    eps=epsilon, 
                                                    delta=delta,
                                                    task='dis')
                g = estimate_cpdag(skel_graph=g, sep_set=sep_set)
                end = time.time()
                run_time['em'].append(end-start)
                f1['em'].append(cal_f1(g.edges, g_answer.edges))
                privacy['em'].append(eps)
                indep_test_numbers['em'].append(test_number)

    return run_time, f1, privacy, indep_test_numbers

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('--iter', type=int, default=5)
    parser.add_argument('--delta', type=float, default=1e-3)
    parser.add_argument('--alg', default='all')
    args = parser.parse_args()

    datasets = ['asia', 'cancer', 'earthquake', 'survey', 'sachs', 'child', 'alarm']
    delta = 1e-3
    epss = np.logspace(-2, 1.5, num=21, base=10)
    metrics = {'priv-pc': {'run_time': {}, 'f1': {}, 'privacy': {}, 'indep_test_number': {}},
               'svt': {'run_time': {}, 'f1': {}, 'privacy': {}, 'indep_test_number': {}},
               'em': {'run_time': {}, 'f1': {}, 'privacy': {}, 'indep_test_number': {}}}
    for i in metrics:
        for j in metrics[i]:
            metrics[i][j]['avg'] = {}
            metrics[i][j]['std'] = {}
            for eps in epss:
                metrics[i][j]['avg'][eps] = -1.0
                metrics[i][j]['std'][eps] = -1.0

    count = 0
    if args.dataset in datasets:
        for eps in epss:
            count += 1
            print('-------------The eps number--------------')
            print(count, eps)
            run_time, f1, privacy, indep_test_numbers = eval(args.dataset, args.iter, epsilon=eps, delta=delta, alpha=ALPHA, option=args.alg)

            metrics['priv-pc']['run_time']['avg'][eps] = np.mean(run_time['priv-pc'])
            metrics['priv-pc']['run_time']['std'][eps] = np.std(run_time['priv-pc'])
            metrics['priv-pc']['f1']['avg'][eps] = np.mean(f1['priv-pc'])
            metrics['priv-pc']['f1']['std'][eps] = np.std(f1['priv-pc'])
            metrics['priv-pc']['privacy']['avg'][eps] = np.mean(privacy['priv-pc'])
            metrics['priv-pc']['privacy']['std'][eps] = np.std(privacy['priv-pc'])

            metrics['priv-pc']['indep_test_number']['avg'][eps] = indep_test_numbers['priv-pc']

            metrics['svt']['run_time']['avg'][eps] = np.mean(run_time['svt'])
            metrics['svt']['run_time']['std'][eps] = np.std(run_time['svt'])
            metrics['svt']['f1']['avg'][eps] = np.mean(f1['svt'])
            metrics['svt']['f1']['std'][eps] = np.std(f1['svt'])
            metrics['svt']['privacy']['avg'][eps] = np.mean(privacy['svt'])
            metrics['svt']['privacy']['std'][eps] = np.std(privacy['svt'])

            metrics['svt']['indep_test_number']['avg'][eps] = indep_test_numbers['svt']

            metrics['em']['run_time']['avg'][eps] = np.mean(run_time['em'])
            metrics['em']['run_time']['std'][eps] = np.std(run_time['em'])
            metrics['em']['f1']['avg'][eps] = np.mean(f1['em'])
            metrics['em']['f1']['std'][eps] = np.std(f1['em'])
            metrics['em']['privacy']['avg'][eps] = np.mean(privacy['em'])
            metrics['em']['privacy']['std'][eps] = np.std(privacy['em'])

            metrics['em']['indep_test_number']['avg'][eps] = indep_test_numbers['em']

        f = open(pkl_template%(args.dataset), 'wb')
        pickle.dump(metrics, f)
        f.close()

    else:

        print('Invalid dataset')
