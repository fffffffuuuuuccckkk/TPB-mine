import torch


def sparsemax(input, dim=-1):
    """
    Sparsemax activation.

    Args:
        input: logits with any shape.
        dim: dimension to normalize.
    Returns:
        Tensor with the same shape as input, sparse and summing to 1 along dim.
    """
    x = input - input.max(dim=dim, keepdim=True).values
    zs = torch.sort(x, dim=dim, descending=True).values
    range_shape = [1] * x.dim()
    range_shape[dim] = x.size(dim)
    rhos = torch.arange(1, x.size(dim) + 1, device=x.device, dtype=x.dtype).view(range_shape)
    cssv = zs.cumsum(dim)
    support = 1 + rhos * zs > cssv
    k = support.sum(dim=dim, keepdim=True).clamp_min(1)
    tau = (cssv.gather(dim, k.long() - 1) - 1) / k.to(x.dtype)
    return torch.clamp(x - tau, min=0.0)


def as_bool(value):
    """Convert bool/int/string config values without treating '0' as True."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in ["1", "true", "yes", "y", "on"]
    return bool(value)
