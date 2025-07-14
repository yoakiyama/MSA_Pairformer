# Adapted from https://github.com/yoakiyama/openfold/blob/main/openfold/utils/chunk_utils.py

import torch
from typing import Callable, Dict, Optional, Tuple, Any
from functools import partial

def dict_map(fn, dic, leaf_type):
    new_dict = {}
    for k, v in dic.items():
        if type(v) is dict:
            new_dict[k] = dict_map(fn, v, leaf_type)
        else:
            new_dict[k] = tree_map(fn, v, leaf_type)

    return new_dict

def spec_dict_map(fn_d, dic, leaf_type):
    new_dict = {}
    for k, v in dic.items():
        if type(v) is dict:
            new_dict[k] = dict_map(fn_d[k], v, leaf_type)
        else:
            new_dict[k] = tree_map(fn_d[k], v, leaf_type)

    return new_dict

def tree_map(fn, tree, leaf_type):
    if isinstance(tree, dict) and isinstance(fn, dict):
        return spec_dict_map(fn, tree, leaf_type)
    elif isinstance(tree, dict):
        return dict_map(fn, tree, leaf_type)
    elif isinstance(tree, list):
        return [tree_map(fn, x, leaf_type) for x in tree]
    elif isinstance(tree, tuple):
        return tuple([tree_map(fn, x, leaf_type) for x in tree])
    elif isinstance(tree, leaf_type):
        return fn(tree)
    else:
        raise ValueError(f"Tree of type {type(tree)} not supported")
tensor_tree_map = partial(tree_map, leaf_type=torch.Tensor)

def _fetch_dims(tree):
    shapes = []
    tree_type = type(tree)
    if tree_type is dict:
        for v in tree.values():
            shapes.extend(_fetch_dims(v))
    elif tree_type is list or tree_type is tuple:
        for t in tree:
            shapes.extend(_fetch_dims(t))
    elif tree_type is torch.Tensor:
        shapes.append(tree.shape)
    else:
        raise ValueError("Not supported")

    return shapes

def chunk_layer(
    layer: Callable,
    inputs: Dict[str, Any],
    chunk_size: int,
    no_batch_dims: int,
    low_mem: bool = False,
    _out: Any = None,
    _add_into_out: bool = False,
    select_chunk_fn_d: Optional[Dict[str, Callable]] = None,
    orig_batch_dims: Optional[Dict[str, Tuple]] = None,
    flat_batch_dim: Optional[int] = None,
    og_batch_dim: Optional[Tuple] = None,
) -> Any:
    if not (len(inputs) > 0):
        raise ValueError("Must provide at least one input")

    if orig_batch_dims is None:
        orig_batch_dims = {}
        initial_dims = [shape[:no_batch_dims] for shape in _fetch_dims(inputs)]
        for k in inputs.keys():
            orig_batch_dims[k] = tuple([max(s) for s in zip(*initial_dims)])
        og_batch_dim = tuple([max(s) for s in zip(*initial_dims)])

    def _prep_inputs(t, batch_dim):
            if(not low_mem):
                if not sum(t.shape[:no_batch_dims]) == no_batch_dims:
                    t = t.expand(batch_dim + t.shape[no_batch_dims:])
                t = t.reshape(-1, *t.shape[no_batch_dims:])
            else:
                t = t.expand(batch_dim + t.shape[no_batch_dims:])
            return t
    prep_inputs_fn_d = {}
    for k in inputs.keys():
        prep_inputs_fn_d[k] = partial(_prep_inputs, batch_dim=orig_batch_dims[k])

    prepped_inputs = tensor_tree_map(prep_inputs_fn_d, inputs)
    prepped_outputs = None
    if(_out is not None):
        reshape_fn = lambda t: t.view([-1] + list(t.shape[no_batch_dims:]))
        prepped_outputs = tensor_tree_map(reshape_fn, _out)

    if flat_batch_dim is None:
        flat_batch_dim = 1
        for d in orig_batch_dims:
            flat_batch_dim *= orig_batch_dims[d][0]

    no_chunks = flat_batch_dim // chunk_size + (
        flat_batch_dim % chunk_size != 0
    )

    i = 0
    out = prepped_outputs
    for _ in range(no_chunks):
        # Chunk the input
        if(not low_mem):
            if select_chunk_fn_d is None:
                select_chunk = (
                    lambda t: t[i : i + chunk_size] if t.shape[0] != 1 else t
                )
            else:
                select_chunk = {k: partial(select_chunk_fn_d[k], i=i, chunk_size=chunk_size) for k in select_chunk_fn_d.keys()}
        else:
            select_chunk = (
                partial(
                    _chunk_slice, 
                    flat_start=i, 
                    flat_end=min(flat_batch_dim, i + chunk_size), 
                    no_batch_dims=len(orig_batch_dims)
                )
            )

        chunks = tensor_tree_map(select_chunk, prepped_inputs)

        # Run the layer on the chunk
        output_chunk = layer(**chunks)

        # Allocate space for the output
        if out is None:
            allocate = lambda t: t.new_zeros((flat_batch_dim,) + t.shape[1:])
            out = tensor_tree_map(allocate, output_chunk)

        # Put the chunk in its pre-allocated space
        out_type = type(output_chunk)
        if out_type is dict:
            def assign(d1, d2):
                for k, v in d1.items():
                    if type(v) is dict:
                        assign(v, d2[k])
                    else:
                        if(_add_into_out):
                            v[i: i + chunk_size] += d2[k]
                        else:
                            v[i: i + chunk_size] = d2[k]

            assign(out, output_chunk)
        elif out_type is tuple:
            for x1, x2 in zip(out, output_chunk):
                if(_add_into_out):
                    x1[i: i + chunk_size] += x2
                else:
                    x1[i : i + chunk_size] = x2
        elif out_type is torch.Tensor:
            if(_add_into_out):
                out[i: i + chunk_size] += output_chunk
            else:
                out[i: i + chunk_size] = output_chunk
        else:
            raise ValueError("Not supported")

        i += chunk_size

    reshape = lambda t: t.view(og_batch_dim + t.shape[1:])
    out = tensor_tree_map(reshape, out)

    return out