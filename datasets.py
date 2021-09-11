# -*- coding: utf-8 -*-

import numpy as np
import scipy.io as sio
import h5py


def load_3sources():
    
    data = sio.loadmat("./data/3-sources.mat")
    
    X = [data['bbc'].toarray().T,
         data['guardian'].toarray().T,
         data['reuters'].toarray().T]
    y_true = np.squeeze(data['truth'])
    
    return X, y_true


def load_BBCSport():
    
    data = sio.loadmat("./data/bbcsport_2view.mat")
    
    X = data['X'][0]
    y_true = np.squeeze(data['gt'])
    
    for idx in range(X.shape[0]):
        X[idx] = X[idx].toarray()
    
    return X, y_true


def load_Caltech7():
    
    data = sio.loadmat("./data/Caltech101-7.mat")
    
    X = data['X'][0][[4]]
    y_true = np.squeeze(data['Y'])
    
    for idx in range(X.shape[0]):
        X[idx] = X[idx].T.astype(float)
    
    return X, y_true


def load_handwritten():
    
    data = sio.loadmat("./data/handwritten.mat")
    
    X = data['X'][0]
    y_true = np.squeeze(data['Y'])
    
    for idx in range(X.shape[0]):
        X[idx] = X[idx].T.astype(float)
    
    return X, y_true


def load_ORL():
    
    data = sio.loadmat("./data/ORL_mtv.mat")
    
    X = data['X'][0]
    y_true = np.squeeze(data['gt'])
    
    return X, y_true


def load_Reuters():
    
    data = {}
    with h5py.File('./data/reuters_1200.mat', 'r') as f:
        data['data'] = []
        for ref in f['data'][0]:
            data['data'].append(np.array(f[ref]))
        data['labels'] = np.array(f['labels'])
    
    X = data['data']
    y_true = np.squeeze(data['labels'])
    
    return X, y_true


def load_dataset(args):
    if args.dataset == '3sources':
        X, y_true = load_3sources()
    elif args.dataset == 'BBCSport':
        X, y_true = load_BBCSport()
    elif args.dataset == 'Caltech7':
        X, y_true = load_Caltech7()
    elif args.dataset == 'handwritten':
        X, y_true = load_handwritten()
    elif args.dataset == 'ORL':
        X, y_true = load_ORL()
    elif args.dataset == 'Reuters':
        X, y_true = load_Reuters()
    else:
        raise ValueError
    return X, y_true



















