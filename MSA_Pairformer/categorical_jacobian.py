import torch
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
from .dataset import nTokenTypes, prepare_additional_molecule_feats, aa2tok_d, prepare_msa_masks

def get_msa_categorical_jacobian(
    model: torch.nn.Module,
    batch_input: torch.tensor, # One-hot encoded MSA size (S, L, 28 alphabet size)
    mask: torch.tensor,
    msa_mask: torch.tensor,
    model_type: str,
    k: int = 0, # Row index of sequence to compute jacobian over,
    mask_only: bool = False,
    eval_mode: bool = False,
    seq_weights: torch.tensor = None
):
    """
    Computes categorical Jacobian for AF3 MSA module or MSATransformer
    """
    assert model_type in ['AF3', 'MSATransformer']
    if eval_mode:
        model = model.eval()

    if model_type == 'AF3':
        # Add batch dimension to one-hot encoding
        batch_input = batch_input.unsqueeze(0) # size (1, S, L, A)

        # Get other inputs
        additional_molecule_feats = prepare_additional_molecule_feats(batch_input)

        # Query sequence length
        seq_length = batch_input.shape[2]
    else:
        seq_length = batch_input.shape[2] - 1

    # Load to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mask = mask.to(device)
    msa_mask = msa_mask.to(device)
    batch_input = batch_input.to(device)
    if model_type == 'AF3':
        additional_molecule_feats = additional_molecule_feats.to(device)
    model = model.to(device)
    if seq_weights is not None:
        seq_weights = seq_weights.to(device)

    with torch.no_grad():
        # Function to get logits for the first sequence (only the 20 amino acids)
        # MSATransformer (ESM) uses 4-23 as AA tokens
        if model_type == 'AF3':
            aa_idx_offset = 0
            f = lambda x: model(msa = x, mask=mask, msa_mask=msa_mask, 
                                additional_molecule_feats=additional_molecule_feats,
                                seq_weights=seq_weights
                                )["logits"][0, k, :seq_length, :20].cpu().numpy()
        else:
            aa_idx_offset = 4
            f = lambda x: model(x)["logits"][0, k, 1:(seq_length+1), aa_idx_offset:20+aa_idx_offset].cpu().numpy()

        # Get logits for unperturbed MSA
        fx = f(batch_input) # L, 20
        # Initialize Jacobian matrix
        if mask_only:
            fx_h = np.zeros((seq_length, 1, seq_length, 20))
        else:
            fx_h = np.zeros((seq_length, 20, seq_length, 20))
        # For each residue
        res_offset = 0 if model_type == 'AF3' else 1 # ESM has a BOS token at the beginning
        for n in tqdm(range(seq_length)):
            # For all 20 mutations
            if mask_only:
                msa_h = torch.clone(batch_input)
                msa_h[0, k, n] = aa2tok_d['MASK']
                fx_h[n, 0] = f(msa_h)
            else:
                for a in range(20):
                    msa_h = torch.clone(batch_input)
                    msa_h[0, k, n+res_offset] = a + aa_idx_offset
                    fx_h[n, a] = f(msa_h)
        return fx - fx_h

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
    



