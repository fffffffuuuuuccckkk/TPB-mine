import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import DataLoader
from datasets import traffic_dataset
from utils import *
import argparse
import yaml
import time
from tqdm import tqdm
import sys
sys.path.append('./model')
sys.path.append('./model/TSFormer')
sys.path.append('./model/Meta_Models')
from meta_patch import *
from reconstruction import *
from TSmodel import *
# from maml_model import *
from rep_model_final import *
from EAGT import SourceEvidenceCache, SourceStructureCache
from EAGT.debug_utils import dump_evidence_csv
from EAGT.sagt_debug_utils import dump_sagt_csv
from CRCT.debug_utils import dump_crct_csv, dump_relation_usage
from pathlib import Path
import random


parser = argparse.ArgumentParser(description='TPB')
parser.add_argument('--config_filename', '--config_file', default='./configs/config_pems.yaml', type=str,
                        help='Configuration filename for restoring the model.')
parser.add_argument('--test_dataset', default='pems-bay', type=str)
parser.add_argument('--train_epochs', default=200, type=int)
parser.add_argument('--finetune_epochs', default=120,type=int)
parser.add_argument('--lr',default=1e-3,type=float)
parser.add_argument('--update_step', default=5,type=int)

parser.add_argument('--seed',default=7,type=int)
parser.add_argument('--data_list', default='chengdu_shenzhen_metr',type=str)
parser.add_argument('--target_days', default=3,type=int)
parser.add_argument('--patch_encoder', default='pattern', type=str)
parser.add_argument('--gpu', default=0, type = int)
parser.add_argument('--sim', default='cosine', type = str)
parser.add_argument('--K', default=10, type = int)
parser.add_argument('--STmodel',default='GWN',type=str)
parser.add_argument('--his_num',default=288,type=int)
parser.add_argument('--use_eagt', default=0, type=int)
parser.add_argument('--eagt_mode', default='edge_v1', type=str, choices=['edge_v1', 'edge_subgraph_v2', 'full_v3'])
parser.add_argument('--eagt_debug', default=0, type=int)
parser.add_argument('--eagt_dry_run', default=0, type=int)
parser.add_argument('--eagt_dump_dir', default='./save/eagt_debug', type=str)
parser.add_argument('--eagt_candidate_topk', default=20, type=int)
parser.add_argument('--eagt_candidate_method', default='corr', type=str, choices=['corr', 'lagcorr', 'corr_lagcorr'])
parser.add_argument('--eagt_include_self_loop', default=0, type=int)
parser.add_argument('--eagt_cache_dir', default='./save/eagt_cache', type=str)
parser.add_argument('--eagt_rebuild_cache', default=0, type=int)
parser.add_argument('--eagt_source_topk_per_node', default=20, type=int)
parser.add_argument('--eagt_max_source_edges', default=200000, type=int)
parser.add_argument('--eagt_retrieval_topk', default=8, type=int)
parser.add_argument('--eagt_chunk_size', default=4096, type=int)
parser.add_argument('--eagt_gamma', default=0.0, type=float)
parser.add_argument('--eagt_row_softmax', default=1, type=int)
parser.add_argument('--eagt_sparse_topk', default=20, type=int)
parser.add_argument('--eagt_sparse_loss_weight', default=0.0, type=float)
parser.add_argument('--eagt_evidence_loss_weight', default=0.0, type=float)
parser.add_argument('--debug_max_batches', default=-1, type=int)
parser.add_argument('--eagt_dump_every', default=0, type=int)
parser.add_argument('--eagt_dump_top_edges', default=20, type=int)
parser.add_argument('--eagt_random_evidence', default=0, type=int)
parser.add_argument('--use_crct', default=0, type=int)
parser.add_argument('--crct_mode', default='v1', type=str, choices=['v1', 'v2_relation_kd', 'v3_concept'])
parser.add_argument('--crct_debug', default=0, type=int)
parser.add_argument('--crct_dry_run', default=0, type=int)
parser.add_argument('--crct_dump_dir', default='./save/crct_debug', type=str)
parser.add_argument('--crct_rho', default=0.0, type=float)
parser.add_argument('--crct_candidate_topk', default=20, type=int)
parser.add_argument('--crct_candidate_method', default='corr', type=str, choices=['corr', 'lagcorr', 'corr_lagcorr', 'dense'])
parser.add_argument('--crct_include_self_loop', default=0, type=int)
parser.add_argument('--crct_sparse_topk', default=20, type=int)
parser.add_argument('--crct_row_softmax', default=1, type=int)
parser.add_argument('--crct_node_encoder', default='tcn', type=str, choices=['mlp', 'tcn', 'gru'])
parser.add_argument('--crct_hidden_dim', default=64, type=int)
parser.add_argument('--crct_relation_dim', default=64, type=int)
parser.add_argument('--crct_dropout', default=0.1, type=float)
parser.add_argument('--crct_num_relations', default=8, type=int)
parser.add_argument('--crct_attribution', default='sparsemax', type=str, choices=['softmax', 'sparsemax', 'entmax15'])
parser.add_argument('--crct_temperature', default=1.0, type=float)
parser.add_argument('--crct_use_unknown', default=1, type=int)
parser.add_argument('--crct_knownness_method', default='entropy', type=str, choices=['maxlogit', 'entropy', 'mlp'])
parser.add_argument('--crct_unknown_floor', default=0.0, type=float)
parser.add_argument('--crct_sparse_loss_weight', default=0.0, type=float)
parser.add_argument('--crct_sharp_loss_weight', default=0.0, type=float)
parser.add_argument('--crct_balance_loss_weight', default=0.0, type=float)
parser.add_argument('--crct_consistency_loss_weight', default=0.0, type=float)
parser.add_argument('--crct_relation_kd_weight', default=0.0, type=float)
parser.add_argument('--crct_unknown_reg_weight', default=0.0, type=float)
parser.add_argument('--crct_dump_every', default=0, type=int)
parser.add_argument('--crct_dump_top_edges', default=20, type=int)
parser.add_argument('--use_sagt', default=0, type=int)
parser.add_argument('--sagt_debug', default=0, type=int)
parser.add_argument('--sagt_cache_dir', default='./save/sagt_cache', type=str)
parser.add_argument('--sagt_rebuild_cache', default=0, type=int)
parser.add_argument('--sagt_use_lowrank', default=1, type=int)
parser.add_argument('--sagt_lowrank_rank', default=8, type=int)
parser.add_argument('--sagt_lowrank_source', default='corr_lagcorr', type=str, choices=['corr', 'lagcorr', 'corr_lagcorr'])
parser.add_argument('--sagt_lowrank_abs', default=1, type=int)
parser.add_argument('--sagt_lowrank_row_softmax', default=1, type=int)
parser.add_argument('--sagt_use_source_roles', default=1, type=int)
parser.add_argument('--sagt_role_dim', default=8, type=int)
parser.add_argument('--sagt_role_iters', default=80, type=int)
parser.add_argument('--sagt_role_lr', default=0.05, type=float)
parser.add_argument('--sagt_role_eps', default=1e-8, type=float)
parser.add_argument('--sagt_role_max_nodes', default=800, type=int)
parser.add_argument('--sagt_role_source_matrix', default='adj_corr', type=str, choices=['adj', 'corr', 'adj_corr'])
parser.add_argument('--sagt_role_nonnegative', default=1, type=int)
parser.add_argument('--sagt_use_spectral_signature', default=1, type=int)
parser.add_argument('--sagt_spectral_rank', default=16, type=int)
parser.add_argument('--sagt_spectral_moments', default=4, type=int)
parser.add_argument('--sagt_spectral_tau', default=1.0, type=float)
parser.add_argument('--sagt_alpha_lowrank', default=0.3, type=float)
parser.add_argument('--sagt_beta_src_role', default=0.4, type=float)
parser.add_argument('--sagt_gamma_eagt', default=0.2, type=float)
parser.add_argument('--sagt_delta_res', default=0.1, type=float)
parser.add_argument('--sagt_sparse_topk', default=20, type=int)
parser.add_argument('--sagt_sparse_loss_weight', default=0.0, type=float)
parser.add_argument('--sagt_rank_loss_weight', default=0.0, type=float)
parser.add_argument('--sagt_res_loss_weight', default=0.0, type=float)
parser.add_argument('--sagt_spec_loss_weight', default=0.0, type=float)
parser.add_argument('--sagt_dump_every', default=0, type=int)
parser.add_argument('--sagt_dump_top_edges', default=30, type=int)
parser.add_argument('--sagt_dump_dir', default='./save/sagt_debug', type=str)
parser.add_argument('--enable_checkpoint', default=None, type=int)
parser.add_argument('--resume', default=None, type=int)
parser.add_argument('--resume_path', default=None, type=str)
parser.add_argument('--save_every', default=None, type=int)
parser.add_argument('--save_best_only', default=None, type=int)
parser.add_argument('--checkpoint_dir', default=None, type=str)
parser.add_argument('--checkpoint_prefix', default=None, type=str)
parser.add_argument('--overwrite_checkpoint', default=None, type=int)
args = parser.parse_args()

args.new=1
seed = args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.set_default_dtype(torch.float32)

IB_DEFAULTS = {
    "use_pattern_ib": False,
    "use_meta_ib": False,
    "pattern_ib_weight": 0.0,
    "pattern_ib_prior": 0.05,
    "pattern_ib_temperature": 1.0,
    "pattern_ib_gate": "gumbel_sigmoid",
    "pattern_ib_topk": 0,
    "pattern_ib_eps": 1e-8,
    "meta_ib_weight": 0.0,
    "meta_ib_dim": 32,
    "meta_ib_prior": "standard_normal",
    "meta_ib_use_label": True,
    "meta_ib_detach_encoder": True,
    "meta_ib_modulate_pattern_query": True,
    "ib_save_suffix": "",
    "ib_save_dir": "./save/ib_runs",
    "enable_checkpoint": True,
    "resume": False,
    "resume_path": "",
    "save_every": 1,
    "save_best_only": False,
    "checkpoint_dir": "./save/ib_runs",
    "checkpoint_prefix": "tpb_ib",
    "overwrite_checkpoint": False
}

EAGT_DEFAULTS = {
    "use_eagt": 0,
    "eagt_mode": "edge_v1",
    "eagt_debug": 0,
    "eagt_dry_run": 0,
    "eagt_dump_dir": "./save/eagt_debug",
    "eagt_candidate_topk": 20,
    "eagt_candidate_method": "corr",
    "eagt_include_self_loop": 0,
    "eagt_cache_dir": "./save/eagt_cache",
    "eagt_rebuild_cache": 0,
    "eagt_source_topk_per_node": 20,
    "eagt_max_source_edges": 200000,
    "eagt_retrieval_topk": 8,
    "eagt_chunk_size": 4096,
    "eagt_gamma": 0.0,
    "eagt_row_softmax": 1,
    "eagt_sparse_topk": 20,
    "eagt_sparse_loss_weight": 0.0,
    "eagt_evidence_loss_weight": 0.0,
    "debug_max_batches": -1,
    "eagt_dump_every": 0,
    "eagt_dump_top_edges": 20,
    "eagt_random_evidence": 0
}

SAGT_DEFAULTS = {
    "use_sagt": 0,
    "sagt_debug": 0,
    "sagt_cache_dir": "./save/sagt_cache",
    "sagt_rebuild_cache": 0,
    "sagt_use_lowrank": 1,
    "sagt_lowrank_rank": 8,
    "sagt_lowrank_source": "corr_lagcorr",
    "sagt_lowrank_abs": 1,
    "sagt_lowrank_row_softmax": 1,
    "sagt_use_source_roles": 1,
    "sagt_role_dim": 8,
    "sagt_role_iters": 80,
    "sagt_role_lr": 0.05,
    "sagt_role_eps": 1e-8,
    "sagt_role_max_nodes": 800,
    "sagt_role_source_matrix": "adj_corr",
    "sagt_role_nonnegative": 1,
    "sagt_use_spectral_signature": 1,
    "sagt_spectral_rank": 16,
    "sagt_spectral_moments": 4,
    "sagt_spectral_tau": 1.0,
    "sagt_alpha_lowrank": 0.3,
    "sagt_beta_src_role": 0.4,
    "sagt_gamma_eagt": 0.2,
    "sagt_delta_res": 0.1,
    "sagt_sparse_topk": 20,
    "sagt_sparse_loss_weight": 0.0,
    "sagt_rank_loss_weight": 0.0,
    "sagt_res_loss_weight": 0.0,
    "sagt_spec_loss_weight": 0.0,
    "sagt_dump_every": 0,
    "sagt_dump_top_edges": 30,
    "sagt_dump_dir": "./save/sagt_debug"
}

CRCT_DEFAULTS = {
    "use_crct": 0,
    "crct_mode": "v1",
    "crct_debug": 0,
    "crct_dry_run": 0,
    "crct_dump_dir": "./save/crct_debug",
    "crct_rho": 0.0,
    "crct_candidate_topk": 20,
    "crct_candidate_method": "corr",
    "crct_include_self_loop": 0,
    "crct_sparse_topk": 20,
    "crct_row_softmax": 1,
    "crct_node_encoder": "tcn",
    "crct_hidden_dim": 64,
    "crct_relation_dim": 64,
    "crct_dropout": 0.1,
    "crct_num_relations": 8,
    "crct_attribution": "sparsemax",
    "crct_temperature": 1.0,
    "crct_use_unknown": 1,
    "crct_knownness_method": "entropy",
    "crct_unknown_floor": 0.0,
    "crct_sparse_loss_weight": 0.0,
    "crct_sharp_loss_weight": 0.0,
    "crct_balance_loss_weight": 0.0,
    "crct_consistency_loss_weight": 0.0,
    "crct_relation_kd_weight": 0.0,
    "crct_unknown_reg_weight": 0.0,
    "crct_dump_every": 0,
    "crct_dump_top_edges": 20
}


def as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in ["1", "true", "yes", "y", "on"]
    return bool(value)


def get_optional_config(config, key, default):
    ib_cfg = config.get("ib", {}) or {}
    ckpt_cfg = config.get("checkpoint", {}) or {}
    if key in config:
        return config[key]
    if key in ib_cfg:
        return ib_cfg[key]
    if key in ckpt_cfg:
        return ckpt_cfg[key]
    return default


def build_ib_args(config):
    ib_args = {key: get_optional_config(config, key, value) for key, value in IB_DEFAULTS.items()}
    if ib_args["ib_save_suffix"] and ib_args["checkpoint_prefix"] == IB_DEFAULTS["checkpoint_prefix"]:
        ib_args["checkpoint_prefix"] = "{}{}".format(ib_args["checkpoint_prefix"], ib_args["ib_save_suffix"])
    if ("checkpoint_dir" not in config and
            "checkpoint_dir" not in (config.get("ib", {}) or {}) and
            "checkpoint_dir" not in (config.get("checkpoint", {}) or {})):
        ib_args["checkpoint_dir"] = ib_args.get("ib_save_dir", IB_DEFAULTS["ib_save_dir"])
    return ib_args


def _arg_was_set(name):
    opt = "--{}".format(name)
    return any(a == opt or a.startswith(opt + "=") for a in sys.argv[1:])


def apply_checkpoint_arg_overrides(ckpt_args, args):
    for key in [
        "enable_checkpoint", "resume", "resume_path", "save_every",
        "save_best_only", "checkpoint_dir", "checkpoint_prefix",
        "overwrite_checkpoint"
    ]:
        if _arg_was_set(key):
            ckpt_args[key] = getattr(args, key)
    for key in ["enable_checkpoint", "resume", "save_best_only", "overwrite_checkpoint"]:
        ckpt_args[key] = bool(ckpt_args[key])
    return ckpt_args


def build_eagt_args(config, args):
    eagt_cfg = config.get("eagt", {}) or {}
    eagt_args = {}
    for key, default in EAGT_DEFAULTS.items():
        if key in config:
            value = config[key]
        elif key in eagt_cfg:
            value = eagt_cfg[key]
        else:
            value = default
        if _arg_was_set(key):
            value = getattr(args, key)
        eagt_args[key] = value
    return eagt_args


def build_sagt_args(config, args):
    sagt_cfg = config.get("sagt", {}) or {}
    eagt_cfg = config.get("eagt", {}) or {}
    sagt_args = {}
    for key, default in SAGT_DEFAULTS.items():
        if key in config:
            value = config[key]
        elif key in sagt_cfg:
            value = sagt_cfg[key]
        elif key in eagt_cfg:
            value = eagt_cfg[key]
        else:
            value = default
        if _arg_was_set(key):
            value = getattr(args, key)
        sagt_args[key] = value
    for key in [
        "use_sagt", "sagt_debug", "sagt_rebuild_cache", "sagt_use_lowrank",
        "sagt_lowrank_abs", "sagt_lowrank_row_softmax",
        "sagt_use_source_roles", "sagt_role_nonnegative",
        "sagt_use_spectral_signature"
    ]:
        sagt_args[key] = int(as_bool(sagt_args[key]))
    return sagt_args


def build_crct_args(config, args):
    crct_cfg = config.get("crct", {}) or {}
    crct_args = {}
    for key, default in CRCT_DEFAULTS.items():
        if key in config:
            value = config[key]
        elif key in crct_cfg:
            value = crct_cfg[key]
        else:
            value = default
        if _arg_was_set(key):
            value = getattr(args, key)
        crct_args[key] = value
    for key in [
        "use_crct", "crct_debug", "crct_dry_run", "crct_include_self_loop",
        "crct_row_softmax", "crct_use_unknown"
    ]:
        crct_args[key] = int(as_bool(crct_args[key]))
    return crct_args


def configure_eagt_checkpoint_defaults(ckpt_args, eagt_args, target_city):
    if not bool(eagt_args.get("use_eagt", 0)):
        return ckpt_args
    if ckpt_args.get("checkpoint_prefix", IB_DEFAULTS["checkpoint_prefix"]) == IB_DEFAULTS["checkpoint_prefix"]:
        gamma = _fmt_float_for_name(eagt_args.get("eagt_gamma", 0.0))
        ckpt_args["checkpoint_prefix"] = "tpb_eagt_g{}_top{}_ret{}".format(
            gamma,
            int(eagt_args.get("eagt_candidate_topk", 20)),
            int(eagt_args.get("eagt_retrieval_topk", 8))
        )
    if ckpt_args.get("checkpoint_dir", IB_DEFAULTS["checkpoint_dir"]) == IB_DEFAULTS["checkpoint_dir"]:
        ckpt_args["checkpoint_dir"] = "./save/eagt_runs"
    return ckpt_args


def configure_crct_checkpoint_defaults(ckpt_args, crct_args, target_city):
    if not as_bool(crct_args.get("use_crct", 0)):
        return ckpt_args
    if ckpt_args.get("checkpoint_prefix", IB_DEFAULTS["checkpoint_prefix"]) == IB_DEFAULTS["checkpoint_prefix"]:
        rho = _fmt_float_for_name(crct_args.get("crct_rho", 0.0))
        ckpt_args["checkpoint_prefix"] = "tpb_crct_r{}_rel{}_top{}".format(
            rho,
            int(crct_args.get("crct_num_relations", 8)),
            int(crct_args.get("crct_candidate_topk", 20))
        )
    if ckpt_args.get("checkpoint_dir", IB_DEFAULTS["checkpoint_dir"]) == IB_DEFAULTS["checkpoint_dir"]:
        ckpt_args["checkpoint_dir"] = "./save/crct_runs"
    return ckpt_args


def configure_sagt_checkpoint_defaults(ckpt_args, sagt_args, target_city):
    if not as_bool(sagt_args.get("use_sagt", 0)):
        return ckpt_args
    current_prefix = ckpt_args.get("checkpoint_prefix", IB_DEFAULTS["checkpoint_prefix"])
    if current_prefix == IB_DEFAULTS["checkpoint_prefix"] or str(current_prefix).startswith("tpb_eagt"):
        ckpt_args["checkpoint_prefix"] = "tpb_sagt_r{}_role{}_top{}".format(
            int(sagt_args.get("sagt_lowrank_rank", 8)),
            int(sagt_args.get("sagt_role_dim", 8)),
            int(sagt_args.get("sagt_sparse_topk", 20))
        )
    current_dir = ckpt_args.get("checkpoint_dir", IB_DEFAULTS["checkpoint_dir"])
    if current_dir == IB_DEFAULTS["checkpoint_dir"] or str(current_dir).endswith("eagt_runs"):
        ckpt_args["checkpoint_dir"] = "./save/sagt_runs"
    return ckpt_args



def _safe_name(x):
    return str(x).replace("/", "_").replace(" ", "").replace(",", "_")


def build_eagt_cache_path(eagt_args, source_cities, target_city):
    source_name = "_".join([_safe_name(c) for c in source_cities])
    target_name = _safe_name(target_city)
    return Path(eagt_args["eagt_cache_dir"]) / "source_{}_target_{}.pt".format(source_name, target_name)


def build_sagt_cache_path(sagt_args, source_cities, target_city):
    source_name = "_".join([_safe_name(c) for c in source_cities])
    target_name = _safe_name(target_city)
    return Path(sagt_args["sagt_cache_dir"]) / "source_{}_target_{}_role{}.pt".format(
        source_name, target_name, int(sagt_args["sagt_role_dim"])
    )


def _collect_source_data_adj(source_dataset, source_cities):
    source_data = {}
    source_adj = {}
    for city in source_cities:
        if city in source_dataset.x_list:
            source_data[city] = source_dataset.x_list[city]
        if city in source_dataset.data_args and "adjacency_matrix_path" in source_dataset.data_args[city]:
            source_adj[city] = np.load(source_dataset.data_args[city]["adjacency_matrix_path"])
        elif city in source_dataset.A_list:
            source_adj[city] = source_dataset.A_list[city]
    return source_data, source_adj


def build_or_load_eagt_cache(source_dataset, source_cities, target_city, eagt_args, device):
    cache_path = build_eagt_cache_path(eagt_args, source_cities, target_city)
    cache = SourceEvidenceCache(eagt_args["eagt_cache_dir"], device="cpu")
    if cache_path.exists() and not bool(eagt_args["eagt_rebuild_cache"]):
        cache.load(cache_path)
        print("[EAGT] loaded source evidence cache: {}".format(cache_path))
    else:
        source_data = {}
        source_adj = {}
        for city in source_cities:
            if city in source_dataset.x_list:
                source_data[city] = source_dataset.x_list[city]
            if city in source_dataset.data_args and "adjacency_matrix_path" in source_dataset.data_args[city]:
                source_adj[city] = np.load(source_dataset.data_args[city]["adjacency_matrix_path"])
            elif city in source_dataset.A_list:
                source_adj[city] = source_dataset.A_list[city]
        if len(source_data) == 0:
            raise ValueError("[EAGT] cannot build source evidence cache: no source city data found")
        cache.build_from_source_data(source_data, source_adj, args=eagt_args)
        cache.save(cache_path)
        print("[EAGT] built source evidence cache: {}".format(cache_path))
    cache.to(device)
    feats = cache.get_features()
    print("[EAGT] use_eagt={}, mode={}, gamma={}, candidate_topk={}, retrieval_topk={}".format(
        eagt_args["use_eagt"], eagt_args["eagt_mode"], eagt_args["eagt_gamma"],
        eagt_args["eagt_candidate_topk"], eagt_args["eagt_retrieval_topk"]
    ))
    print("[EAGT] cache path={}, source evidences={}, feature dim={}".format(
        cache_path, feats.shape[0], feats.shape[1] if feats.dim() == 2 else 0
    ))
    return cache, cache_path


def build_or_load_sagt_cache(source_dataset, source_cities, target_city, sagt_args, device):
    cache_path = build_sagt_cache_path(sagt_args, source_cities, target_city)
    cache = SourceStructureCache(sagt_args["sagt_cache_dir"], device="cpu")
    if cache_path.exists() and not as_bool(sagt_args["sagt_rebuild_cache"]):
        cache.load(cache_path)
        print("[SAGT] loaded source structure cache: {}".format(cache_path))
    else:
        source_data, source_adj = _collect_source_data_adj(source_dataset, source_cities)
        if len(source_data) == 0:
            raise ValueError("[SAGT] cannot build source structure cache: no source city data found")
        cache.build_from_source_data(source_data, source_adj, args=sagt_args)
        cache.save(cache_path)
        print("[SAGT] built source structure cache: {}".format(cache_path))
    cache.to(device)
    sig = cache.get_spectral_signatures()
    print("[SAGT] use_sagt={}, lowrank_rank={}, role_dim={}, alpha/beta/gamma/delta={}/{}/{}/{}".format(
        sagt_args["use_sagt"], sagt_args["sagt_lowrank_rank"], sagt_args["sagt_role_dim"],
        sagt_args["sagt_alpha_lowrank"], sagt_args["sagt_beta_src_role"],
        sagt_args["sagt_gamma_eagt"], sagt_args["sagt_delta_res"]
    ))
    print("[SAGT] source structure cache path={}".format(cache_path))
    print("[SAGT] source cities={}".format(cache.get_city_names()))
    print("[SAGT] spectral signature shape={}".format(tuple(sig.shape) if sig is not None else None))
    print("[SAGT] role_B count={}".format(len(cache.get_role_B())))
    return cache, cache_path


def maybe_dump_eagt(rep_model, source_cache, eagt_args, epoch, batch_id=0, force=False):
    if not bool(eagt_args["use_eagt"]) or not bool(eagt_args["eagt_debug"]):
        return None
    dump_every = int(eagt_args.get("eagt_dump_every", 0))
    if not force and dump_every <= 0:
        return None
    if not force and (batch_id + 1) % dump_every != 0:
        return None
    debug_dict = getattr(rep_model.model, "latest_eagt_debug", None)
    if not debug_dict:
        return None
    dump_dir = Path(eagt_args["eagt_dump_dir"])
    csv_path = dump_dir / "eagt_epoch{}_batch{}.csv".format(epoch, batch_id)
    dump_evidence_csv(debug_dict, source_cache.get_metadata(), csv_path, top_edges=eagt_args["eagt_dump_top_edges"])
    print("[EAGT] epoch={}, batch={}".format(epoch, batch_id))
    print("[EAGT] A_original stats: {}".format(debug_dict.get("A_original_stats", {})))
    print("[EAGT] A_eagt stats: {}".format(debug_dict.get("A_eagt_stats", {})))
    print("[EAGT] A_final stats: {}".format(debug_dict.get("A_final_stats", {})))
    print("[EAGT] target candidate edges={}".format(debug_dict.get("target_candidate_count", 0)))
    print("[EAGT] evidence csv: {}".format(csv_path))
    return csv_path


def maybe_dump_crct(rep_model, crct_args, epoch, batch_id=0, force=False):
    if not as_bool(crct_args.get("use_crct", 0)) or not as_bool(crct_args.get("crct_debug", 0)):
        return None
    dump_every = int(crct_args.get("crct_dump_every", 0))
    if not force and dump_every <= 0:
        return None
    if not force and (batch_id + 1) % dump_every != 0:
        return None
    debug_dict = getattr(rep_model.model, "latest_crct_debug", None)
    if not debug_dict:
        return None
    dump_dir = Path(crct_args["crct_dump_dir"])
    csv_path = dump_dir / "crct_epoch{}_batch{}.csv".format(epoch, batch_id)
    usage_path = dump_dir / "crct_relation_usage_epoch{}_batch{}.csv".format(epoch, batch_id)
    dump_crct_csv(debug_dict, csv_path, top_edges=crct_args["crct_dump_top_edges"])
    dump_relation_usage(debug_dict, usage_path)
    print("[CRCT] epoch={}, batch={}".format(epoch, batch_id))
    print("[CRCT] A_original stats: {}".format(debug_dict.get("A_original_stats", {})))
    print("[CRCT] A_crct stats: {}".format(debug_dict.get("A_crct_stats", {})))
    print("[CRCT] A_final stats: {}".format(debug_dict.get("A_final_stats", {})))
    print("[CRCT] knownness stats: {}".format(debug_dict.get("knownness_stats", {})))
    print("[CRCT] relation_usage: {}".format(
        [round(float(x), 6) for x in debug_dict.get("relation_usage").detach().cpu().tolist()]
        if debug_dict.get("relation_usage", None) is not None else []
    ))
    print("[CRCT] target candidate edges={}".format(debug_dict.get("target_candidate_count", 0)))
    print("[CRCT] attribution csv: {}".format(csv_path))
    print("[CRCT] relation usage csv: {}".format(usage_path))
    return csv_path


def maybe_dump_sagt(rep_model, source_structure_cache, sagt_args, epoch, batch_id=0, force=False):
    if not as_bool(sagt_args.get("use_sagt", 0)) or not as_bool(sagt_args.get("sagt_debug", 0)):
        return None
    dump_every = int(sagt_args.get("sagt_dump_every", 0))
    if not force and dump_every <= 0:
        return None
    if not force and (batch_id + 1) % dump_every != 0:
        return None
    debug_dict = getattr(rep_model.model, "latest_sagt_debug", None)
    if not debug_dict:
        return None
    dump_dir = Path(sagt_args["sagt_dump_dir"])
    csv_path = dump_dir / "sagt_epoch{}_batch{}.csv".format(epoch, batch_id)
    dump_sagt_csv(debug_dict, source_structure_cache, csv_path, top_edges=sagt_args["sagt_dump_top_edges"])
    print("[SAGT] epoch={}, batch={}".format(epoch, batch_id))
    print("[SAGT] A_lowrank stats: {}".format(debug_dict.get("A_lowrank_stats", {})))
    print("[SAGT] A_src_role stats: {}".format(debug_dict.get("A_src_role_stats", {})))
    print("[SAGT] A_eagt stats: {}".format(debug_dict.get("A_eagt_stats", {})))
    print("[SAGT] A_res stats: {}".format(debug_dict.get("A_res_stats", {})))
    print("[SAGT] A_sagt stats: {}".format(debug_dict.get("A_sagt_stats", {})))
    print("[SAGT] A_final stats: {}".format(debug_dict.get("A_final_stats", {})))
    print("[SAGT] source city weights: {}".format(
        [round(float(x), 6) for x in debug_dict.get("source_city_weight").detach().cpu().tolist()]
        if debug_dict.get("source_city_weight", None) is not None else []
    ))
    print("[SAGT] target candidate edges={}".format(debug_dict.get("target_candidate_count", 0)))
    print("[SAGT] attribution csv: {}".format(csv_path))
    return csv_path

def _fmt_float_for_name(x):
    """Convert float-like values to filename-safe short strings."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x).replace(".", "p")
    if x == 0:
        return "0"
    s = "{:.0e}".format(x) if abs(x) < 1e-3 else "{:g}".format(x)
    return s.replace("-", "m").replace(".", "p").replace("+", "")


def build_param_tag(ib_args, args):
    """Build a short checkpoint name tag from current IB/search parameters."""
    tags = []

    if ib_args.get("use_pattern_ib", False):
        tags.extend([
            "pib",
            "pw{}".format(_fmt_float_for_name(ib_args.get("pattern_ib_weight", 0.0))),
            "pp{}".format(_fmt_float_for_name(ib_args.get("pattern_ib_prior", 0.0))),
            "pg{}".format(str(ib_args.get("pattern_ib_gate", "none"))),
            "pt{}".format(_fmt_float_for_name(ib_args.get("pattern_ib_temperature", 1.0))),
        ])
        topk = int(ib_args.get("pattern_ib_topk", 0))
        if topk > 0:
            tags.append("topk{}".format(topk))

    if ib_args.get("use_meta_ib", False):
        tags.extend([
            "mib",
            "mw{}".format(_fmt_float_for_name(ib_args.get("meta_ib_weight", 0.0))),
            "md{}".format(int(ib_args.get("meta_ib_dim", 0))),
            "mod{}".format(int(bool(ib_args.get("meta_ib_modulate_pattern_query", False)))),
        ])

    tags.extend([
        "K{}".format(args.K),
        "seed{}".format(args.seed),
    ])

    return "_".join(tags) if tags else "base"

import re

def find_latest_checkpoint(checkpoint_dir, checkpoint_prefix, target_city):
    """
    Auto find the checkpoint with the largest epoch index under current prefix.

    Expected filenames:
        {checkpoint_prefix}_{target_city}_epoch{i}.pt
        {checkpoint_prefix}_{target_city}_last.pt
    """
    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.exists():
        return None

    pattern = re.compile(
        r"^{}_{city}_(?:finetune_)?epoch(\d+)(?:_[^.]+)?\.pt$".format(
            re.escape(checkpoint_prefix),
            city=re.escape(target_city)
        )
    )

    meta_candidates = []
    finetune_candidates = []

    for glob_pat, bucket in [
        ("{}_{}_epoch*.pt".format(checkpoint_prefix, target_city), meta_candidates),
        ("{}_{}_finetune_epoch*.pt".format(checkpoint_prefix, target_city), finetune_candidates),
    ]:
        for ckpt_path in checkpoint_dir.glob(glob_pat):
            match = pattern.match(ckpt_path.name)
            if match:
                epoch = int(match.group(1))
                bucket.append((epoch, ckpt_path.stat().st_mtime, ckpt_path))

    if finetune_candidates:
        finetune_candidates.sort(key=lambda x: (x[0], x[1]))
        return finetune_candidates[-1][2]

    if meta_candidates:
        meta_candidates.sort(key=lambda x: (x[0], x[1]))
        return meta_candidates[-1][2]

    last_candidates = list(checkpoint_dir.glob("{}_{}_last*.pt".format(checkpoint_prefix, target_city)))
    if last_candidates:
        last_candidates.sort(key=lambda p: p.stat().st_mtime)
        return last_candidates[-1]

    return None


def peek_checkpoint_stage(path):
    ckpt = torch.load(path, map_location="cpu")
    return ckpt.get("stage", "meta"), int(ckpt.get("epoch", -1)), ckpt.get("best_metric", None)


def resolve_resume_path(ib_args, target_city):
    """
    If resume_path is given, use it.
    Otherwise, automatically find the latest checkpoint of current prefix.
    """
    resume_path = str(ib_args.get("resume_path", "")).strip()

    if resume_path:
        return resume_path

    auto_path = find_latest_checkpoint(
        ib_args["checkpoint_dir"],
        ib_args["checkpoint_prefix"],
        target_city
    )

    if auto_path is None:
        raise FileNotFoundError(
            "resume=True but resume_path is empty, and no checkpoint was found under "
            "dir='{}' with prefix='{}' and target_city='{}'.".format(
                ib_args["checkpoint_dir"],
                ib_args["checkpoint_prefix"],
                target_city
            )
        )

    print("INFO: resume_path is empty. Auto found latest checkpoint: {}".format(auto_path))
    return str(auto_path)
# since historical 1 day data is used to generate metaknowledge
print(time.strftime('%Y-%m-%d %H:%M:%S'), "Forecasting target_days = {}".format(args.target_days - 1))

if __name__ == '__main__':
    if torch.cuda.is_available():
        args.device = torch.device('cuda:0')
        print("INFO: GPU : {}".format(args.gpu))
    else:
        args.device = torch.device('cpu')
        print("INFO: CPU")

    with open(args.config_filename) as f:
        config = yaml.safe_load(f)
    config = config or {}
    ib_args = build_ib_args(config)
    eagt_args = build_eagt_args(config, args)
    sagt_args = build_sagt_args(config, args)
    crct_args = build_crct_args(config, args)
    ib_args = apply_checkpoint_arg_overrides(ib_args, args)


    print(config)
    args.data_list = config['model']['STnet']['data_list']
    args.batch_size = config['task']['maml']['batch_size']
    args.test_dataset = config['task']['maml']['test_dataset']
    args.K = config['model']['STnet']['K']
    data_args, task_args, model_args = config['data'], config['task'], config['model']
    data_list = get_data_list(args.data_list)
    source_city_list = list(data_list)
    print("INFO: train on {}. test on {}.".format(data_list, args.test_dataset))
    PatchFSL_cfg = {
        'data_list' : args.data_list,
        'sim': args.sim,
        'K' : args.K,
        'patch_encoder': args.patch_encoder,
        'base_dir': Path(sys.path[0]),
        'device':args.device
    }
    if ib_args["use_pattern_ib"] or ib_args["use_meta_ib"]:
        param_tag = build_param_tag(ib_args, args)
        base_prefix = str(ib_args.get("checkpoint_prefix", "tpb_ib"))
        if param_tag not in base_prefix:
            ib_args["checkpoint_prefix"] = "{}_{}".format(base_prefix, param_tag)
        print("INFO: checkpoint_prefix = {}".format(ib_args["checkpoint_prefix"]))
    PatchFSL_cfg_print = dict(PatchFSL_cfg)
    PatchFSL_cfg.update({'test_dataset': args.test_dataset, 'config': config})
    PatchFSL_cfg.update(ib_args)
    PatchFSL_cfg.update(eagt_args)
    PatchFSL_cfg.update(sagt_args)
    PatchFSL_cfg.update(crct_args)
    ib_enabled = ib_args["use_pattern_ib"] or ib_args["use_meta_ib"]
    eagt_enabled = as_bool(eagt_args["use_eagt"])
    sagt_enabled = as_bool(sagt_args["use_sagt"])
    crct_enabled = as_bool(crct_args["use_crct"])
    ib_args = configure_eagt_checkpoint_defaults(ib_args, eagt_args, args.test_dataset)
    ib_args = configure_crct_checkpoint_defaults(ib_args, crct_args, args.test_dataset)
    ib_args = configure_sagt_checkpoint_defaults(ib_args, sagt_args, args.test_dataset)
    PatchFSL_cfg.update(ib_args)
    dry_run_enabled = (
        (eagt_enabled and as_bool(eagt_args.get("eagt_dry_run", 0))) or
        (crct_enabled and as_bool(crct_args.get("crct_dry_run", 0)))
    )
    checkpoint_enabled = as_bool(ib_args["enable_checkpoint"]) and (ib_enabled or eagt_enabled or crct_enabled or sagt_enabled) and not dry_run_enabled
    if ib_enabled:
        print("INFO: IB enabled. Pattern IB={}, Meta IB={}".format(ib_args["use_pattern_ib"], ib_args["use_meta_ib"]))
    if checkpoint_enabled:
        print("INFO: checkpoint_prefix = {}".format(ib_args["checkpoint_prefix"]))
        print("INFO: checkpoint_dir = {}".format(ib_args["checkpoint_dir"]))
    if crct_enabled:
        print("[CRCT] use_crct={}, mode={}, rho={}, num_relations={}, relation_dim={}, attribution={}".format(
            crct_args["use_crct"], crct_args["crct_mode"], crct_args["crct_rho"],
            crct_args["crct_num_relations"], crct_args["crct_relation_dim"],
            crct_args["crct_attribution"]
        ))
        if eagt_enabled:
            print("[CRCT] Both use_eagt and use_crct are enabled. CRCT v1 skips EAGT inside PatchFSL.")
    if sagt_enabled:
        print("[SAGT] use_sagt={}, lowrank_rank={}, role_dim={}, alpha/beta/gamma/delta={}/{}/{}/{}".format(
            sagt_args["use_sagt"], sagt_args["sagt_lowrank_rank"], sagt_args["sagt_role_dim"],
            sagt_args["sagt_alpha_lowrank"], sagt_args["sagt_beta_src_role"],
            sagt_args["sagt_gamma_eagt"], sagt_args["sagt_delta_res"]
        ))

    ## dataset
    source_dataset = traffic_dataset(data_args, task_args['maml'], list(source_city_list), "source_train", test_data=args.test_dataset)
    ## check source dataset

    for data in source_dataset.data_list:
        print("source dataset has {}. X : {}, y : {}".format(data,source_dataset.x_list[data].shape,source_dataset.y_list[data].shape))

    source_evidence_cache = None
    if eagt_enabled and (not crct_enabled or sagt_enabled):
        source_evidence_cache, eagt_cache_path = build_or_load_eagt_cache(
            source_dataset,
            source_city_list,
            args.test_dataset,
            eagt_args,
            args.device
        )
        PatchFSL_cfg["source_evidence_cache"] = source_evidence_cache

    source_structure_cache = None
    if sagt_enabled:
        source_structure_cache, sagt_cache_path = build_or_load_sagt_cache(
            source_dataset,
            source_city_list,
            args.test_dataset,
            sagt_args,
            args.device
        )
        PatchFSL_cfg["source_structure_cache"] = source_structure_cache

    finetune_dataset = traffic_dataset(data_args, task_args['maml'], list(source_city_list), 'target_maml', test_data=args.test_dataset)
    test_dataset = traffic_dataset(data_args, task_args['maml'], list(source_city_list), 'test', test_data=args.test_dataset)
    print(data_args, task_args, model_args, PatchFSL_cfg if (ib_enabled or eagt_enabled or crct_enabled or sagt_enabled) else PatchFSL_cfg_print, args.STmodel)
    rep_model = STRep(data_args, task_args, model_args, PatchFSL_cfg, args.STmodel)
    best_loss = 9999999999999.0
    best_model = None
    start_epoch = 0
    best_metric = None
    skip_meta_train = False
    finetune_resume_path = None
    if ib_args["resume"]:
        if not checkpoint_enabled:
            raise ValueError("resume=True requires use_pattern_ib, use_meta_ib, use_eagt, use_crct, or use_sagt with enable_checkpoint=True.")

        resume_path = resolve_resume_path(ib_args, args.test_dataset)
        resume_stage, resume_epoch, best_metric = peek_checkpoint_stage(resume_path)
        if resume_stage == "finetune":
            skip_meta_train = True
            finetune_resume_path = resume_path
            print("INFO: Found finetune checkpoint at epoch {}. Skip meta-training and resume finetuning.".format(resume_epoch))
        else:
            start_epoch, best_metric = load_checkpoint(
                resume_path,
                rep_model,
                rep_model.meta_optim
            )

            rep_model.current_epoch = start_epoch

            print("INFO: Resumed from checkpoint: {}".format(resume_path))
            print("INFO: Continue meta-training from epoch {}".format(start_epoch))
    if best_metric is None:
        best_metric = 9999999999999.0
    ## train on big dataset
    rep_tasknum = task_args['maml']['task_num']

    for i in range(start_epoch, task_args['maml']['train_epochs']) if not skip_meta_train else []:
        length = source_dataset.__len__()
        # length=40
        print('----------------------')
        time_1 = time.time()

        data_spt = []
        matrix_spt = []
        data_qry = []
        matrix_qry = []

        idx = 0
        active_rep_tasknum = rep_tasknum
        if int(eagt_args["debug_max_batches"]) > 0:
            active_rep_tasknum = min(active_rep_tasknum, int(eagt_args["debug_max_batches"]))
        for jj in range(active_rep_tasknum):
            data_i, A = source_dataset[idx]
            data_spt.append(data_i)
            matrix_spt.append(A)
            idx+=1

            data_i, A = source_dataset[idx]
            data_qry.append(data_i)
            matrix_qry.append(A)
            idx+=1

        model_loss ,mse_loss, rmse_loss, mae_loss, mape_loss = rep_model.meta_train_revise(data_spt, matrix_spt, data_qry, matrix_qry)

        print('Epochs {}/{}'.format(i,task_args['maml']['train_epochs']))
        print('in meta-training   Unnormed MSE : {:.5f}, RMSE : {:.5f}, MAE : {:.5f}, MAPE: {:.5f}, reconstruction Loss : {:.5f}.'.format(mse_loss, rmse_loss, mae_loss,mape_loss,model_loss))
        if ib_enabled:
            print("IB loss pred={:.5f}, pattern={:.5f}, meta={:.5f}, total={:.5f}".format(
                rep_model.last_ib_log.get("pred_loss", 0.0),
                rep_model.last_ib_log.get("pattern_ib_loss", 0.0),
                rep_model.last_ib_log.get("meta_ib_loss", 0.0),
                rep_model.last_ib_log.get("total_loss", 0.0)
            ))
        if crct_enabled:
            print("CRCT loss sparse={:.5f}, sharp={:.5f}, balance={:.5f}, kd={:.5f}, unknown={:.5f}, total={:.5f}".format(
                rep_model.last_crct_log.get("crct_sparse_loss", 0.0),
                rep_model.last_crct_log.get("crct_sharp_loss", 0.0),
                rep_model.last_crct_log.get("crct_balance_loss", 0.0),
                rep_model.last_crct_log.get("crct_relation_kd_loss", 0.0),
                rep_model.last_crct_log.get("crct_unknown_reg_loss", 0.0),
                rep_model.last_crct_log.get("total_loss", 0.0)
            ))
        if sagt_enabled:
            print("SAGT loss sparse={:.5f}, rank={:.5f}, res={:.5f}, spec={:.5f}, total={:.5f}".format(
                rep_model.last_sagt_log.get("sagt_sparse_loss", 0.0),
                rep_model.last_sagt_log.get("sagt_rank_loss", 0.0),
                rep_model.last_sagt_log.get("sagt_res_loss", 0.0),
                rep_model.last_sagt_log.get("sagt_spec_loss", 0.0),
                rep_model.last_sagt_log.get("total_loss", 0.0)
            ))
        if checkpoint_enabled:
            checkpoint_dir = Path(ib_args["checkpoint_dir"])
            checkpoint_prefix = ib_args["checkpoint_prefix"]
            target_city = args.test_dataset
            metric = float(mae_loss)
            is_best = metric < best_metric
            if is_best:
                best_metric = metric
                save_checkpoint(
                    checkpoint_dir / '{}_{}_best.pt'.format(checkpoint_prefix, target_city),
                    rep_model.checkpoint_state(i, rep_model.meta_optim, config, best_metric, stage="meta"),
                    overwrite=ib_args["overwrite_checkpoint"]
                )
            if (not ib_args["save_best_only"]) and ((i + 1) % max(1, int(ib_args["save_every"])) == 0):
                state = rep_model.checkpoint_state(i, rep_model.meta_optim, config, best_metric, stage="meta")
                save_checkpoint(checkpoint_dir / '{}_{}_last.pt'.format(checkpoint_prefix, target_city), state, overwrite=ib_args["overwrite_checkpoint"])
                save_checkpoint(checkpoint_dir / '{}_{}_epoch{}.pt'.format(checkpoint_prefix, target_city, i), state, overwrite=ib_args["overwrite_checkpoint"])
        print("This epoch cost {:.3}s.".format(time.time() - time_1))
        if eagt_enabled and not crct_enabled:
            maybe_dump_eagt(
                rep_model,
                source_evidence_cache,
                eagt_args,
                epoch=i,
                batch_id=0,
                force=as_bool(eagt_args["eagt_dry_run"])
            )
            if as_bool(eagt_args["eagt_dry_run"]):
                print("[EAGT] dry_run finished.")
                sys.exit(0)
        if crct_enabled:
            maybe_dump_crct(
                rep_model,
                crct_args,
                epoch=i,
                batch_id=0,
                force=as_bool(crct_args["crct_dry_run"])
            )
            if as_bool(crct_args["crct_dry_run"]):
                print("[CRCT] dry_run finished.")
                sys.exit(0)
        if sagt_enabled:
            maybe_dump_sagt(
                rep_model,
                source_structure_cache,
                sagt_args,
                epoch=i,
                batch_id=0,
                force=False
            )
    rep_model.finetuning(
        finetune_dataset,
        test_dataset,
        task_args['maml']['finetune_epochs'],
        resume_path=finetune_resume_path
    )
