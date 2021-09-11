# -*- coding: utf-8 -*-

import argparse

from datasets import load_dataset
from model import MCML
from utils import save_model

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        description='Multi-view co-similarity metric learning',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--alpha', default=None, type=float)
    parser.add_argument('--beta', default=1e-5, type=float)
    parser.add_argument('--tol_err', default=1e-5, type=float)
    parser.add_argument('--maxIter', default=1000, type=int)
    parser.add_argument('--random_state', default=0, type=int)
    parser.add_argument('--dev', default=None, type=str)
    parser.add_argument('--save_dir', default='./results')
    args = parser.parse_args()
    
    X, _ = load_dataset(args)
    
    mcml = MCML(X)
    
    mcml.fit(args)
    
    save_model(mcml, args)



















