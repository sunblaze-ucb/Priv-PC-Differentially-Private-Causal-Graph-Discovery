# Priv-PC: Differentially Private Causal Graph Discovery

This is the code accompanying the Neurips 2020 paper ["Towards practical differentially private causal graph discovery"](https://arxiv.org/abs/2006.08598). The code for the original pc algorithm is borrowed from this excellent [repo](https://github.com/keiichishima/pcalg).

# Overview

Priv-PC is designed for differentially private causal graph discovery. Priv-PC leverages sieve-and-examine mechanism to augment PC algorithm with differential privacy. Intuitively, Priv-PC uses sparse vector technique to sieve out "unsignificant" queries while using substantial privacy budget to carefully examine "significant" ones.

# Prequisities

- Python 3.6.10
- R 3.4.4

# Reproduce the evaluation results

First, download all dependencies by running `pip install -r requirements.txt`.

The evaluation can be reproduced using `python eval.py name_of_dataset`.
