import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

def plot_contact_map(
    cons,
    L=1,
    minsep=24,
    cutoffs=[None],
    ss=[5],
    cc=["gray"],
    f=None,
    ax=None, 
    vmin=None, 
    vmax=None
):
    max_L = 0
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(10, 10))

    #### Get upper triangle matrix for heatmap ####
    prot_len = cons[0].shape[0]
    upper = np.full((prot_len, prot_len), np.nan)
    for i in range(prot_len):
        for j in range(i, prot_len):  # i <= j
            if j - i >= minsep:
                upper[i, j] = cons[1][i, j]
            else:
                upper[i, j] = np.nan
    im = ax.imshow(upper, origin="upper", cmap="Greys", rasterized=True, vmin=vmin, vmax=vmax)

    #### Plot lower triangle matrix using top-L or ground truth contacts ####

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
            top = int(con.shape[0] * L)
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
            alpha = 0.8
        if label == "Ground truth":
            # ax.scatter(filtered_i[vals_sort_idx], filtered_j[vals_sort_idx], c=c, s=s, rasterized=True, label=label)
            ax.scatter(filtered_i[vals_sort_idx], filtered_j[vals_sort_idx], c=c, s=s, rasterized=True, edgecolors='none')#, label=label)
            ax.scatter([-1], [-1], c=c, s=50, label=label, rasterized=True, edgecolors='none')
        
        if n == 1 and len(cons) > 1:
            # For the second contact map, highlight false positives (where ground truth is 0)
            # We need to match indices between the two contact maps
            if filtered_i.size > 0:  # Make sure we have pairs to check
                # Get values from first contact map at same positions
                first_map_vals = cons[0][filtered_i[vals_sort_idx], filtered_j[vals_sort_idx]]
                bad = first_map_vals == 0
                
                # Plot false positives
                # ax.scatter(filtered_i[vals_sort_idx][bad], filtered_j[vals_sort_idx][bad], c="red", s=s, alpha=alpha, rasterized=True)
                label = "False positives"
                ax.scatter(filtered_i[vals_sort_idx][bad], filtered_j[vals_sort_idx][bad], c="red", s=s, alpha=alpha, rasterized=True, edgecolors='none')#, label=label)
                ax.scatter([-1], [-1], c="red", s=50, label="False positives", rasterized=True, edgecolors='none')
            label = "True positives"
            ax.scatter(filtered_i[vals_sort_idx][first_map_vals == 1], filtered_j[vals_sort_idx][first_map_vals == 1], c=c, s=s, alpha=alpha, rasterized=True, edgecolors='none')#, label=label)
            ax.scatter([-1], [-1], c=c, s=50, label=label, rasterized=True, edgecolors='none')

    ax.set_xlim(0, max_L)
    ax.set_ylim(0, max_L)
    ax.set_aspect('equal')
    ax.set_ylim(ax.get_ylim()[::-1])
    ax.set_xticks([])
    ax.set_yticks([])
    
    return f, ax

def eval_hetero_oligomer(
    cons,
    chain_break_pos,
    L=1,
    monomer_minsep=24,
    cutoffs=[None],
    ss=[5],
    cc=['gray'],
    f=None,
    ax=None,
    vmax=None,
    vmin=0,
    monomer_p_at_k=False
    ):
    # Get complex length
    n = cons[0].shape[0]
    complex_length = cons[0].shape[0]
    #### Get upper triangle matrix for heatmap ####
    upper = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i, n):  # i <= j
            if j - i >= monomer_minsep:
                upper[i, j] = cons[1][i, j]
            else:
                upper[i, j] = np.nan
    im = ax.imshow(upper, origin="upper", cmap="Greys", rasterized=True, vmin=vmin, vmax=vmax)

    ##### Plot ground truth (n = 0) and predictions #####
    # Monomer
    precision_d = {}
    total_interface_contacts = None
    monomer_top_d = {}
    for i, (con, cutoff, s, c) in enumerate(zip(cons, cutoffs, ss, cc)):
        # Iterate over both chains separately
        for chain_idx in range(2):
            chain_start_idx = chain_idx * chain_break_pos
            if chain_idx == 0:
                chain_end_idx = (chain_idx + 1) * chain_break_pos
            else:
                chain_end_idx = complex_length
            sub_con = deepcopy(con[chain_start_idx:chain_end_idx, chain_start_idx:chain_end_idx])
            triu_idx = np.triu_indices_from(sub_con, 1)
            mask = np.abs(triu_idx[0] - triu_idx[1]) >= monomer_minsep
            filtered_i, filtered_j = triu_idx[0][mask], triu_idx[1][mask]
            vals = sub_con[filtered_i, filtered_j]
            if (i == 0) and monomer_p_at_k:
                monomer_top_d[chain_idx] = vals.sum()
            elif (i == 0) and not monomer_p_at_k:
                monomer_top_d[chain_idx] = sub_con.shape[0] * L
            if cutoff is None:
                top = monomer_top_d[chain_idx]
                if len(vals) > top:  # Make sure we have enough values
                    use_cutoff = np.sort(vals)[::-1][top]
                else:
                    use_cutoff = np.min(vals) if len(vals) > 0 else 0
            else:
                use_cutoff = cutoff
            if np.all((con == 0) | (con == 1)):
                vals_sort_idx = vals == 1
                label = "Ground truth"
            else:
                vals_sort_idx = vals > use_cutoff
                label = "True positives"
            if chain_idx > 0:
                label = None
            if i == 1 and len(cons) > 1:
                # For the second contact map, highlight false positives (where ground truth is 0)
                # We need to match indices between the two contact maps
                if filtered_i.size > 0:  # Make sure we have pairs to check
                    # Get values from first contact map at same positions
                    first_map_vals = cons[0][filtered_i[vals_sort_idx] + chain_start_idx, filtered_j[vals_sort_idx] + chain_start_idx]
                    bad = first_map_vals == 0
                    
                    # Plot false positives
                    ax.scatter(filtered_i[vals_sort_idx][bad] + chain_start_idx, filtered_j[vals_sort_idx][bad] + chain_start_idx, c="red", s=s, alpha=0.8, edgecolors='none', rasterized=True)
                    label = None if chain_idx > 0 else "False positives"
                    ax.scatter([-1], [-1], c="red", s=70, label=label, edgecolors='none', rasterized=True)
                precision_d[f"chain_{chain_idx}"] = (len(bad) - bad.sum()) / len(bad)
                # Plot true positives
                label = None if chain_idx > 0 else "True positives"
                ax.scatter([-1], [-1], c=c, s=70, label=label, edgecolors='none', rasterized=True)
                ax.scatter(filtered_i[vals_sort_idx][first_map_vals == 1] + chain_start_idx, filtered_j[vals_sort_idx][first_map_vals == 1] + chain_start_idx, c=c, s=s, alpha=0.8, edgecolors='none', rasterized=True)
            else:
                # Plot ground truth
                label = None if chain_idx > 0 else "Ground truth"
                ax.scatter([-1], [-1], c=c, s=70, label=label, edgecolors='none', rasterized=True)
                ax.scatter(filtered_i[vals_sort_idx] + chain_start_idx, filtered_j[vals_sort_idx] + chain_start_idx, c=c, s=s, edgecolors='none', rasterized=True)

        # Hetero-oligomer contacts
        sub_con = deepcopy(con[chain_break_pos:, :chain_break_pos])
        if i == 0:
            total_interface_contacts = sub_con.sum()
        if total_interface_contacts == 0:
            continue
        filtered_i, filtered_j = np.indices(sub_con.shape)
        filtered_i, filtered_j = filtered_i.flatten(), filtered_j.flatten()
        vals = sub_con[filtered_i, filtered_j]
        if cutoff is None:
            hetero_top = total_interface_contacts
            if len(vals) > hetero_top:
                use_cutoff = np.sort(vals)[::-1][hetero_top]
            else:
                use_cutoff = np.min(vals) if len(vals) > 0 else 0
        else:
            use_cutoff = cutoff
        if np.all((sub_con == 0) | (sub_con == 1)):
            vals_sort_idx = vals == 1
            label = "Ground truth"
        else:
            vals_sort_idx = vals > use_cutoff
            label = "True positives"
        if i == 1 and len(cons) > 1:
            # For the second contact map, highlight false positives (where ground truth is 0)
            # We need to match indices between the two contact maps
            if filtered_i.size > 0:  # Make sure we have pairs to check
                # Get values from first contact map at same positions
                gt_sub_con = deepcopy(cons[0][chain_break_pos:, :chain_break_pos])
                first_map_vals = gt_sub_con[filtered_i[vals_sort_idx], filtered_j[vals_sort_idx]]
                bad = first_map_vals == 0
                
                # Plot false positives
                ax.scatter(filtered_j[vals_sort_idx][bad], filtered_i[vals_sort_idx][bad] + chain_break_pos, c="red", s=s, alpha=0.8, edgecolors='none', rasterized=True)
            precision_d[f"interface"] = (len(bad) - bad.sum()) / len(bad)
            # Plot true positives
            ax.scatter(filtered_j[vals_sort_idx][first_map_vals == 1], filtered_i[vals_sort_idx][first_map_vals == 1] + chain_break_pos, c=c, s=s, alpha=0.8, edgecolors='none', rasterized=True)
        else:
            # Plot ground truth
            ax.scatter(filtered_j[vals_sort_idx], filtered_i[vals_sort_idx] + chain_break_pos, c=c, s=s, edgecolors='none', rasterized=True)
    ax.set_xlim(0, cons[0].shape[0])
    ax.set_ylim(0, cons[0].shape[0])
    ax.set_aspect('equal')
    ax.set_ylim(ax.get_ylim()[::-1])
    ax.set_xticks([])
    ax.set_yticks([])
    return f, ax, precision_d