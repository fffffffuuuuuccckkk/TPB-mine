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
        r"^{}_{city}_epoch(\d+)\.pt$".format(
            re.escape(checkpoint_prefix),
            city=re.escape(target_city)
        )
    )

    candidates = []

    for ckpt_path in checkpoint_dir.glob("{}_{}_epoch*.pt".format(checkpoint_prefix, target_city)):
        match = pattern.match(ckpt_path.name)
        if match:
            epoch = int(match.group(1))
            candidates.append((epoch, ckpt_path))

    if candidates:
        candidates.sort(key=lambda x: x[0])
        return candidates[-1][1]

    last_path = checkpoint_dir / "{}_{}_last.pt".format(checkpoint_prefix, target_city)
    if last_path.exists():
        return last_path

    return None


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


    print(config)
    args.data_list = config['model']['STnet']['data_list']
    args.batch_size = config['task']['maml']['batch_size']
    args.test_dataset = config['task']['maml']['test_dataset']
    args.K = config['model']['STnet']['K']
    data_args, task_args, model_args = config['data'], config['task'], config['model']
    data_list = get_data_list(args.data_list)
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
    ib_enabled = ib_args["use_pattern_ib"] or ib_args["use_meta_ib"]
    if ib_enabled:
        print("INFO: IB enabled. Pattern IB={}, Meta IB={}".format(ib_args["use_pattern_ib"], ib_args["use_meta_ib"]))

    ## dataset
    source_dataset = traffic_dataset(data_args, task_args['maml'], data_list, "source_train", test_data=args.test_dataset)
    ## check source dataset

    for data in source_dataset.data_list:
        print("source dataset has {}. X : {}, y : {}".format(data,source_dataset.x_list[data].shape,source_dataset.y_list[data].shape))

    finetune_dataset = traffic_dataset(data_args, task_args['maml'], data_list, 'target_maml', test_data=args.test_dataset)
    test_dataset = traffic_dataset(data_args, task_args['maml'], data_list, 'test', test_data=args.test_dataset)
    print(data_args, task_args, model_args, PatchFSL_cfg if ib_enabled else PatchFSL_cfg_print, args.STmodel)
    rep_model = STRep(data_args, task_args, model_args, PatchFSL_cfg, args.STmodel)
    best_loss = 9999999999999.0
    best_model = None
    start_epoch = 0
    best_metric = None
    if ib_args["resume"]:
        if not ib_enabled:
            raise ValueError("resume=True is only supported for IB checkpoints; enable use_pattern_ib or use_meta_ib.")

        resume_path = resolve_resume_path(ib_args, args.test_dataset)

        start_epoch, best_metric = load_checkpoint(
            resume_path,
            rep_model,
            rep_model.meta_optim
        )

        rep_model.current_epoch = start_epoch
        start_epoch = start_epoch + 1

        print("INFO: Resumed from checkpoint: {}".format(resume_path))
        print("INFO: Continue training from epoch {}".format(start_epoch))
    if best_metric is None:
        best_metric = 9999999999999.0
    ## train on big dataset
    rep_tasknum = task_args['maml']['task_num']

    for i in range(start_epoch, task_args['maml']['train_epochs']):
        length = source_dataset.__len__()
        # length=40
        print('----------------------')
        time_1 = time.time()

        data_spt = []
        matrix_spt = []
        data_qry = []
        matrix_qry = []

        idx = 0
        for jj in range(rep_tasknum):
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
            if ib_args["enable_checkpoint"]:
                checkpoint_dir = Path(ib_args["checkpoint_dir"])
                checkpoint_prefix = ib_args["checkpoint_prefix"]
                target_city = args.test_dataset
                metric = float(mae_loss)
                is_best = metric < best_metric
                if is_best:
                    best_metric = metric
                    save_checkpoint(
                        checkpoint_dir / '{}_{}_best.pt'.format(checkpoint_prefix, target_city),
                        rep_model.checkpoint_state(i, rep_model.meta_optim, config, best_metric),
                        overwrite=ib_args["overwrite_checkpoint"]
                    )
                if (not ib_args["save_best_only"]) and ((i + 1) % max(1, int(ib_args["save_every"])) == 0):
                    state = rep_model.checkpoint_state(i, rep_model.meta_optim, config, best_metric)
                    save_checkpoint(checkpoint_dir / '{}_{}_last.pt'.format(checkpoint_prefix, target_city), state, overwrite=ib_args["overwrite_checkpoint"])
                    save_checkpoint(checkpoint_dir / '{}_{}_epoch{}.pt'.format(checkpoint_prefix, target_city, i), state, overwrite=ib_args["overwrite_checkpoint"])
        print("This epoch cost {:.3}s.".format(time.time() - time_1))
    rep_model.finetuning(finetune_dataset, test_dataset, task_args['maml']['finetune_epochs'])
