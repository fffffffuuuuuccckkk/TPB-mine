import math

import torch


def entropy_loss(attr):
    """Mean attribution entropy for attr [B,E,K]."""
    eps = 1e-8
    return -(attr.clamp_min(eps) * attr.clamp_min(eps).log()).sum(dim=-1).mean()


def balance_loss(attr):
    """KL(mean relation usage || uniform) for attr [B,E,K]."""
    eps = 1e-8
    mean_attr = attr.mean(dim=(0, 1)).clamp_min(eps)
    uniform = torch.full_like(mean_attr, 1.0 / max(1, mean_attr.numel()))
    return (mean_attr * (mean_attr.log() - uniform.log())).sum()


def sparse_loss(A):
    """Mean absolute adjacency weight for A [N,N] or [B,N,N]."""
    return A.abs().mean()


def combine_crct_losses(aux_loss_dict, args):
    """
    Weighted CRCT auxiliary objective.

    All weights default to 0, so this returns a zero scalar unless explicitly enabled.
    """
    if not aux_loss_dict:
        raise ValueError("CRCT aux_loss_dict is empty")
    device = next(iter(aux_loss_dict.values())).device
    dtype = next(iter(aux_loss_dict.values())).dtype
    total = torch.zeros((), device=device, dtype=dtype)
    log_dict = {}
    for name in [
        "crct_sparse_loss",
        "crct_sharp_loss",
        "crct_balance_loss",
        "crct_consistency_loss",
        "crct_relation_kd_loss",
        "crct_unknown_reg_loss",
    ]:
        weight_name = name.replace("_loss", "_loss_weight")
        weight = _get_arg(args, weight_name, 0.0)
        value = aux_loss_dict.get(name, torch.zeros((), device=device, dtype=dtype))
        total = total + float(weight) * value
        log_dict[name] = float(value.detach().item())
    log_dict["crct_total_loss"] = float(total.detach().item())
    return total, log_dict


def _get_arg(args, name, default):
    if args is None:
        return default
    if isinstance(args, dict):
        return args.get(name, default)
    return getattr(args, name, default)
