# -*- coding: utf-8 -*-

import numpy as np
import igraph as ig
import leidenalg as la
import pickle
from pathlib import Path
from scipy.sparse import csr_matrix
from scipy.optimize import linear_sum_assignment


def save_model(mcml, args):
    with open(Path(args.save_dir) / ('model_'+args.dataset+'.pickle'), 'wb') as f:
        pickle.dump(mcml, f)


def load_model(args):
    with open(Path(args.model_dir) / ('model_'+args.dataset+'.pickle'), 'rb') as f:
        model = pickle.load(f)
    return model


def get_igraph_from_adjacency(adjacency, directed=None):
    """Get igraph graph from adjacency matrix."""
    sources, targets = adjacency.nonzero()
    weights = adjacency[sources, targets]
    if isinstance(weights, np.matrix):
        weights = weights.A1
    g = ig.Graph(directed=directed)
    g.add_vertices(adjacency.shape[0])  # this adds adjacency.shape[0] vertices
    g.add_edges(list(zip(sources, targets)))
    try:
        g.es['weight'] = weights
    except KeyError:
        pass
    return g


def leiden_clustering(Z, args):
    g = get_igraph_from_adjacency(csr_matrix(Z))
    part = la.find_partition(
        graph=g,
        partition_type=la.RBConfigurationVertexPartition,
        weights='weight',
        resolution_parameter=args.resolution,
        seed=args.random_state,
    )
    y_pred = np.array(part.membership)
    return y_pred


def best_cluster_fit(y_true, y_pred):
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_sum_assignment(w.max() - w)
    best_fit = []
    for i in range(y_pred.size):
        for j in range(len(ind)):
            if ind[j][0] == y_pred[i]:
                best_fit.append(ind[j][1])
    return best_fit, ind, w


def cluster_acc(y_true, y_pred):
    y_true = np.array(y_true).astype(np.int64)
    y_pred = np.array(y_pred).astype(np.int64)
    _, ind, w = best_cluster_fit(y_true, y_pred)
    return sum([w[ind[0][i], ind[1][i]] for i in range(len(ind[0]))]) * 1.0 / y_pred.size



















