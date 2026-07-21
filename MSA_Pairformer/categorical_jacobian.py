import torch
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
from .dataset import aa2tok_d

def get_contacts_from_coev(x, symm=True, center=True, rm=1, mask_only=False):
    # convert jacobian (L,A,L,A) to contact map (L,L)
    j = x.copy()
    if center:
        if mask_only:
            for i in [l for l in range(4) if l != 1]: j -= j.mean(i,keepdims=True)
        else:
            for i in range(4): j -= j.mean(i,keepdims=True)
    j_fn = np.sqrt(np.square(j).sum((1,3)))
    np.fill_diagonal(j_fn,0)
    j_fn_corrected = do_apc(j_fn, rm=rm)
    if symm:
        j_fn_corrected = (j_fn_corrected + j_fn_corrected.T)/2
    return j_fn_corrected

def do_apc(x, rm=1):
    '''given matrix do apc correction'''
    # trying to remove different number of components
    # rm=0 remove none
    # rm=1 apc
    x = np.copy(x)
    if rm == 0:
        return x
    elif rm == 1:
        a1 = x.sum(0,keepdims=True)
        a2 = x.sum(1,keepdims=True)
        y = x - (a1*a2)/x.sum()
    else:
        # decompose matrix, rm largest(s) eigenvectors
        u,s,v = np.linalg.svd(x)
        y = s[rm:] * u[:,rm:] @ v[rm:,:]
    np.fill_diagonal(y,0)
    return y

def get_contacts_from_pairwise_rep(pairwise_rep: torch.tensor):
    """
    Computes L2-norm of (L, L, d) pairwise representation
    """
    if type(pairwise_rep) == np.ndarray:
        return np.linalg.norm(pairwise_rep, axis=2)
    else:
        return torch.norm(pairwise_rep, p=2, dim=2)

def get_coevolution(X, num_aa_types=23):
    '''given one-hot encoded MSA, return contacts'''
    Y = jax.nn.one_hot(X, num_aa_types)
    N,L,A = Y.shape
    Y_flat = Y.reshape(N,-1)
    # covariance
    c = jnp.cov(Y_flat.T)

    # inverse covariance
    shrink = 4.5/jnp.sqrt(N) * jnp.eye(c.shape[0])
    ic = jnp.linalg.inv(c + shrink)

    # partial correlation coefficient
    ic_diag = jnp.diag(ic)
    pcc = ic / jnp.sqrt(ic_diag[:,None] * ic_diag[None,:])
    
    raw = jnp.sqrt(jnp.square(pcc.reshape(L,A,L,A)[:,:20,:,:20]).sum((1,3)))
    i = jnp.arange(L)
    raw = raw.at[i,i].set(0)
    # do apc
    ap = (raw.sum(0, keepdims=True) * raw.sum(1, keepdims=True)) / raw.sum()
    return (raw - ap).at[i,i].set(0)
    



