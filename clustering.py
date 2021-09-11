# -*- coding: utf-8 -*-

import argparse

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from datasets import load_dataset
from utils import load_model, leiden_clustering, cluster_acc


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        description='Multi-view co-similarity metric learning',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--resolution', type=float, required=True)
    parser.add_argument('--random_state', default=0, type=int)
    parser.add_argument('--model_dir', default='./results')
    args = parser.parse_args()
    
    _, y_true = load_dataset(args)
    
    mcml = load_model(args)
    
    y_pred = leiden_clustering(mcml.get_sample_similarity(), args)
    
    if len(set(y_pred)) < len(set(y_true)):
        print('Note: a larger resoltion is recommended')
    elif len(set(y_pred)) > len(set(y_true)):
        print('Note: a smaller resoltion is recommended')
    print('number of predicted clusters:', len(set(y_pred)))
    print('number of true classes:', len(set(y_true)))
    print('ACC:', cluster_acc(y_true, y_pred))
    print('ARI:', adjusted_rand_score(y_true, y_pred))
    print('NMI:', normalized_mutual_info_score(y_true, y_pred))



















