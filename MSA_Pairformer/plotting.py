import matplotlib.pyplot as plt
import numpy as np

def plot_contact_map(cons, L=1, minsep=24, cutoffs=[None], ss=[5], cc=["gray"], f=None, ax=None):
    max_L = 0
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(10, 10))
    for n, (con, cutoff, s, c) in enumerate(zip(cons,cutoffs,ss,cc)):
        if con.shape[0] > max_L:
            max_L = con.shape[0]
        
        # Get upper triangular indices
        triu_idx = np.triu_indices_from(con, 1)
        
        # Create a mask for contacts that are at least minsep apart
        mask = np.abs(triu_idx[0] - triu_idx[1]) >= minsep
        
        # Apply the mask to the indices
        filtered_i = triu_idx[0][mask]
        filtered_j = triu_idx[1][mask]
        
        # Get values for these filtered indices
        vals = con[filtered_i, filtered_j]
        
        if cutoff is None:
            top = con.shape[0] * L
            # Use the filtered indices directly
            if len(vals) > top:  # Make sure we have enough values
                cutoff = np.sort(vals)[::-1][top]
            else:
                cutoff = np.min(vals) if len(vals) > 0 else 0
        
        if np.all((con == 0) | (con == 1)):
            vals_sort_idx = vals >= cutoff
            label = "Ground truth"
        else:
            vals_sort_idx = vals > cutoff
            label = "True positives"
        
        # Use the filtered indices with the sorting
        if label == "Ground truth":
            alpha = 1
        else:
            alpha = 0.5
        ax.scatter(filtered_i[vals_sort_idx], filtered_j[vals_sort_idx], c=c, s=s, alpha=alpha, rasterized=True)
        ax.scatter(filtered_j[vals_sort_idx], filtered_i[vals_sort_idx], c=c, s=s, alpha=alpha, rasterized=True)
        ax.scatter([-1], [-1], c=c, s=20, label=label, rasterized=True)
        
        if n == 1 and len(cons) > 1:
            # For the second contact map, highlight false positives (where ground truth is 0)
            # We need to match indices between the two contact maps
            if filtered_i.size > 0:  # Make sure we have pairs to check
                # Get values from first contact map at same positions
                first_map_vals = cons[0][filtered_i[vals_sort_idx], filtered_j[vals_sort_idx]]
                bad = first_map_vals == 0
                
                # Plot false positives
                ax.scatter(filtered_i[vals_sort_idx][bad], filtered_j[vals_sort_idx][bad], c="red", s=s, alpha=alpha, rasterized=True)
                ax.scatter(filtered_j[vals_sort_idx][bad], filtered_i[vals_sort_idx][bad], c="red", s=s, alpha=alpha, rasterized=True)
                ax.scatter([-1], [-1], c="red", s=20, label="False positives", rasterized=True)
    
    ax.set_xlim(0, max_L)
    ax.set_ylim(0, max_L)
    ax.set_aspect('equal')
    ax.set_ylim(ax.get_ylim()[::-1])
    ax.set_xticks([])
    ax.set_yticks([])
    
    return f, ax