# -*- coding: utf-8 -*-

from typing import Optional, Tuple

import torch
from torch import Tensor
import numpy as np
from tqdm import trange


def updateW(
    X: Tensor,
    Z: Tensor,
    Y_w: Tensor,
    z_eye: Tensor,
    ones_w: Tensor,
    lamba_w: Tensor,
    beta: float,
    rho: float,
    ceta: float,
) -> Tuple[Tensor, Tensor]:
    
    optimize_zz = torch.matmul(Z, Z.T) + z_eye
    
    L_w_matrix = torch.matmul(torch.matmul(X, 2*optimize_zz - 2*Z), X.T)
        
    L_w = 2*torch.norm(L_w_matrix, 2)+ beta*lamba_w.shape[1]/ceta
    
    M_w = torch.matmul(2*L_w_matrix + beta/ceta, Y_w) - 2*torch.matmul(torch.matmul(X, optimize_zz), X.T) - beta/ceta + lamba_w
    
    W = (Y_w - M_w/L_w) * ones_w
    W = (torch.abs(W) + W) / 2
    W = (W + W.T) / 2
    
    leq3 = torch.sum(W, dim=0) - 1
    
    lamba_w = lamba_w + beta*rho*leq3
    
    return W, lamba_w


def updateZ(
    X: Tensor,
    W: Tensor,
    Y_z: Tensor,
    w_eye: Tensor,
    ones_z: Tensor,
    lamba_z: Tensor,
    alpha: list,
    beta: float,
    rho: float,
    ceta: float,
) -> Tuple[Tensor, Tensor]:
    
    n_views = len(X)
    
    optimize_ww = []
    L_z_matrix = []
    for view in range(n_views):
        optimize_ww.append(torch.matmul(W[view], W[view].T) + w_eye[view])
        L_z_matrix.append(torch.matmul(torch.matmul(X[view].T, 2*optimize_ww[view] - 2*W[view]), X[view]))
    
    L_z_matrix_sum = L_z_matrix[0].clone()
    for view in range(1, n_views):
        L_z_matrix_sum += alpha[view] * L_z_matrix[view]
    
    L_z = 2*torch.norm(L_z_matrix_sum, 2) + beta*lamba_z.shape[1]/ceta
    
    M_z = torch.matmul(2*L_z_matrix[0] + beta/ceta, Y_z) - 2*torch.matmul(torch.matmul(X[0].T, optimize_ww[0]), X[0]) - beta/ceta + lamba_z
    for view in range(1, n_views):
        M_z += 2*alpha[view]*torch.matmul(L_z_matrix[view], Y_z) - 2*alpha[view]*torch.matmul(torch.matmul(X[view].T, optimize_ww[view]), X[view]) 
    
    Z = (Y_z - M_z/L_z)*ones_z
    Z = (torch.abs(Z) + Z)/2
    Z = (Z + Z.T)/2
    
    leq4 = torch.sum(Z, dim=0) - 1
    
    lamba_z = lamba_z + beta*rho*leq4
    
    return Z, lamba_z


def MCML_fit(
    X: list,
    W: list,
    Z: Tensor,
    alpha: list,
    beta: float,
    tol_err: float,
    maxIter: int,
    SS_matrix: Optional[np.ndarray],
    FS_matrix: Optional[list],
    dev: str,
) -> Tuple[Tensor, Tensor, Tensor]:
    
    device = torch.device(dev)
    
    n_views = len(X)
    
    m = []
    n = X[0].shape[1]
    for view in range(n_views):
        m.append(X[view].shape[0])
    
    rho = 0.8
    ceta_prev = 1 / rho
    ceta = 1
    
    func_err = float('inf')
    
    W_prev = []
    Z_prev = Z
    for view in range(n_views):
        W_prev.append(W[view])
    
    lamba_w = []
    lamba_z = torch.zeros(1, n).to(device)
    for view in range(n_views):
        lamba_w.append(torch.zeros(1, m[view]).to(device))
    
    w_eye = []
    z_eye = torch.eye(n).to(device)
    for view in range(n_views):
        w_eye.append(torch.eye(m[view]).to(device))
    
    if SS_matrix is None:
        ones_z = 1 - z_eye
    else:
        ones_z = torch.tensor(SS_matrix, dtype=torch.float32).to(device)
    
    ones_w = []
    if FS_matrix is None:
        for view in range(n_views):
            ones_w.append(1 - w_eye[view])
    else:
        for view in range(n_views):
            ones_w.append(torch.tensor(FS_matrix[view], dtype=torch.float32).to(device))
    
    pbar = trange(maxIter)
    
    for Iter in pbar:
        
        func_err_prev = func_err
        
        Y_iter_value = (ceta*(1-ceta_prev))/ceta_prev
        
        Y_w = []
        Y_z = Z + Y_iter_value*(Z - Z_prev)
        Z_prev = Z
        for view in range(n_views):
            Y_w.append(W[view] + Y_iter_value*(W[view] - W_prev[view]))
            W_prev[view] = W[view]
        
        for view in range(n_views):
            W[view], lamba_w[view] = updateW(
                X=X[view],
                Z=Z,
                Y_w=Y_w[view],
                z_eye=z_eye,
                ones_w=ones_w[view],
                lamba_w=lamba_w[view],
                beta=beta,
                rho=rho,
                ceta=ceta,
            )
        
        Z, lamba_z = updateZ(
            X=X,
            W=W,
            Y_z=Y_z,
            w_eye=w_eye,
            ones_z=ones_z,
            lamba_z=lamba_z,
            alpha=alpha,
            beta=beta,
            rho=rho,
            ceta=ceta,
        )
        
        ceta_prev = ceta
        ceta = 1/(1 - rho + 1/ceta)
        
        func_1_err = []
        func_2_err = []
        func_3_err = []
        func_4_err = []
        for view in range(n_views):
            func_1_err.append(torch.norm(torch.matmul(W[view].T, torch.matmul(X[view], z_eye-Z)), 'fro'))
            func_2_err.append(torch.norm(torch.matmul(X[view], z_eye-Z), 'fro'))
            func_3_err.append(torch.norm(torch.matmul(Z.T, torch.matmul(X[view].T, w_eye[view]-W[view])), 'fro'))
            func_4_err.append(torch.norm(torch.matmul(X[view].T, w_eye[view]-W[view]), 'fro'))
        
        func_err = func_1_err[0] + func_2_err[0] + func_3_err[0] + func_4_err[0]
        for view in range(1, n_views):
            func_err += alpha[view] * (func_1_err[view] + func_2_err[view] + func_3_err[view] + func_4_err[view])
        
        func_err_rel = torch.abs(func_err_prev - func_err) / func_err_prev
        
        pbar.set_postfix_str(f'relative error: {func_err_rel.item():.3e}')
        
        if func_err_rel < tol_err:
            pbar.set_postfix_str(f'relative error: {func_err_rel.item():.3e}, converged!')
            break
        
    return W, Z, func_err


class MCML():
    
    def __init__(self, X):
        
        self.X = X
        self.n_views = len(X)
        self.n_samples = X[0].shape[1]
    
    def fit(self, args):
        
        dev = args.dev
        if dev is None or dev == "cuda":
            if torch.cuda.is_available():
              dev = "cuda"
            else:
              dev = "cpu"
        
        device = torch.device(dev)
        
        rng = torch.Generator()
        rng.manual_seed(args.random_state)
        
        r = []
        m = []
        n = self.n_samples
        W = []
        Z = torch.rand(n, n, generator=rng).to(device)
        for view in range(self.n_views):
            r.append(torch.tensor(self.X[view]).type(dtype=torch.float32).to(device))
            m.append(self.X[view].shape[0])
            W.append(torch.rand(m[view], m[view], generator=rng).to(device))
        
        alpha = np.ones(self.n_views) if args.alpha is None else args.alpha
        X_0_norm = torch.norm(r[0], 'fro')
        for view in range(1, self.n_views):
            alpha[view] *= X_0_norm / torch.norm(r[view], 'fro') / alpha[0]
        alpha[0] = 1
        
        W, Z, err = MCML_fit(
            X=r,
            W=W,
            Z=Z,
            alpha=alpha,
            beta=args.beta,
            tol_err=args.tol_err,
            maxIter=args.maxIter,
            SS_matrix=None,
            FS_matrix=None,
            dev=dev,
        )
        
        for view in range(self.n_views):
            W[view] = W[view].cpu().numpy()
            
        self.W = W
        self.Z = Z.cpu().numpy()
        self.err = err.cpu().numpy()
    
    def get_feature_similarity(self):
        return self.W
    
    def get_sample_similarity(self):
        return self.Z



















