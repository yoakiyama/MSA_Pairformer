import torch
import math
import numpy as np
from scipy.stats import truncnorm
from typing import Literal

# Helper functions
def _prod(nums):
    out = 1
    for n in nums:
        out = out * n
    return out

def _calculate_fan(
    linear_weight_shape,
    fan: Literal["fan_in", "fan_out", "fan_avg"] = "fan_in"
):
    assert fan in ["fan_in", "fan_out", "fan_avg"], "Invalid fan option"
    fan_out, fan_in = linear_weight_shape
    if fan == "fan_in":
        return fan_in
    elif fan == "fan_out":
        return fan_out
    elif fan == "fan_avg":
        return (fan_in + fan_out) / 2

# Initialize weight parameters with ones/zeros
def _init_ones_(M):
    with torch.no_grad():
        M.fill_(1.0)
def _init_zeros_(M):
    with torch.no_grad():
        M.fill_(0.0)

# Truncated normal initialization
def trunc_normal_init_(
    M,
    scale=1.0,
    fan="fan_in"
):
    # Determine parameters for truncated normal distribution
    shape = M.shape
    f = _calculate_fan(shape, fan)
    scale = scale / max(1, f)
    a = -2
    b = 2
    std = math.sqrt(scale) / truncnorm.std(a=a, b=b, loc=0, scale=1)
    size = _prod(shape)
    # Sample from truncated normal distribution
    samples = truncnorm.rvs(a=a, b=b, loc=0, scale=std, size=size)
    # Reshape samples to match the shape of the weights
    samples = np.reshape(samples, shape)
    # Copy samples to the weights
    with torch.no_grad():
        M.copy_(torch.tensor(samples, device=M.device))

# LeCun normal initialization (truncate between -2*std and 2*std)
def lecun_normal_init_(M):
    trunc_normal_init_(M, scale=1.0, fan="fan_in")