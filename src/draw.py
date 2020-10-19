import os 
import sys
import csv
import numpy as np 
import pylab
import argparse
import pickle
import matplotlib
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

pkl_location='../results/pkl/%s.pkl'
fig_location='../results/figure/%s.pdf'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('--metric', default='f1')
    parser.add_argument('--lower', type=float, default=1)
    parser.add_argument('--upper', type=float, default=1e2)
    args = parser.parse_args()

    if args.metric == 'run_time':
        pass

    else:
        epss = np.logspace(-3, 2, 5)
        with open(pkl_location%args.dataset, 'rb') as pickle_file:
            metrics = pickle.load(pickle_file)

        ax = plt.subplot()
        ax.set_xscale("log", nonposx='clip')

        for alg in ['svt']:
            print(metrics[alg]['indep_test_number']['avg'].values())
            avg_list = list(metrics[alg][args.metric]['avg'].values())
            std_list = list(metrics[alg][args.metric]['std'].values())
            privacy_avg_list = list(metrics[alg]['privacy']['avg'].values())

            rm_avg_list = []
            rm_std_list = []
            rm_priv_list = []
            for idx, priv in enumerate(privacy_avg_list):
                if priv < args.lower or priv > args.upper:
                    rm_avg_list.append(avg_list[idx])
                    rm_std_list.append(std_list[idx])
                    rm_priv_list.append(priv)
            for v in rm_avg_list:
                avg_list.remove(v)
            for v in rm_std_list:
                std_list.remove(v)
            for v in rm_priv_list:
                privacy_avg_list.remove(v)
            ax.errorbar(privacy_avg_list, avg_list, yerr=std_list, label=alg, capsize=5, capthick=2, elinewidth=1, linestyle='dashed', marker='o')

        plt.xlabel('Epsilon', fontsize=20)
        plt.ylabel(args.metric, fontsize=20)
        plt.legend()
        plt.grid(True)

        with PdfPages(fig_location%args.dataset) as pdf:
            pdf.savefig(bbox_inches='tight')

if __name__ == '__main__':
    main()
