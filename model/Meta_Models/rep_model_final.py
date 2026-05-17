import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import GATConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import PairTensor, Adj, OptTensor, Size
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)
import time
from copy import deepcopy
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch.nn.utils import weight_norm
import math
import tqdm
import copy

from meta_patch import *
from reconstruction import *
from EAGT import EAGTGraphConstructor, SourceEvidenceCache, SAGTGraphConstructor
from CRCT import CRCTGraphConstructor
from CRCT.sparse_ops import as_bool
import sys
from pathlib import Path
import os
sys.path.append('../TSFormer')
from TSmodel import *
sys.path.append('../../')
from utils import *
import datetime


def safe_checkpoint_path(path, overwrite=False):
    path = Path(path)
    if overwrite or not path.exists():
        return path
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    candidate = path.with_name("{}_{}{}".format(path.stem, stamp, path.suffix))
    idx = 1
    while candidate.exists():
        candidate = path.with_name("{}_{}_{}{}".format(path.stem, stamp, idx, path.suffix))
        idx += 1
    return candidate


def save_checkpoint(path, state, overwrite=False):
    path = safe_checkpoint_path(path, overwrite)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    print("Checkpoint saved: {}".format(path))
    return path


def load_checkpoint(path, model, optimizer=None):
    if not path:
        raise ValueError("resume=True requires resume_path.")
    if not os.path.exists(path):
        raise FileNotFoundError("resume_path not found: {}".format(path))
    ckpt = torch.load(path, map_location=getattr(model, "device", None))
    incompatible = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    missing = len(getattr(incompatible, "missing_keys", []))
    unexpected = len(getattr(incompatible, "unexpected_keys", []))
    print("Loaded with strict=False: missing={}, unexpected={}".format(missing, unexpected))
    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    start_epoch = int(ckpt.get("epoch", -1)) + 1
    best_metric = ckpt.get("best_metric", None)
    print("Resumed from {}, start_epoch={}".format(path, start_epoch))
    return start_epoch, best_metric


class TaskIBEncoder(nn.Module):
    """
    Infers a task-level bottleneck variable from the support set.
    mu: task latent posterior mean.
    logvar: task latent posterior log variance.
    u_tau: sampled task-level bottleneck variable.
    meta_ib_loss: KL(q(u_tau|support) || N(0,I)).
    meta_ib_dim: latent dimension.
    meta_ib_weight: loss weight, applied by STRep.combine_losses.
    """
    def __init__(self, input_dim, meta_ib_dim=32, hidden_dim=64, detach_encoder=True):
        super(TaskIBEncoder, self).__init__()
        self.detach_encoder = detach_encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mu = nn.Linear(hidden_dim, meta_ib_dim)
        self.logvar = nn.Linear(hidden_dim, meta_ib_dim)

    def forward(self, data_i, device, use_label=True, deterministic=False):
        x = torch.tensor(data_i.x, dtype=torch.float32, device=device)
        x_speed = x[..., 0]
        feats = [x_speed.mean(dim=(1, 2)), x_speed.std(dim=(1, 2), unbiased=False)]
        if use_label and hasattr(data_i, "y"):
            y = torch.tensor(data_i.y, dtype=torch.float32, device=device)
            feats.extend([y.mean(dim=tuple(range(1, y.dim()))), y.std(dim=tuple(range(1, y.dim())), unbiased=False)])
        task_summary = torch.stack(feats, dim=-1)
        task_summary = (task_summary - task_summary.mean(dim=-1, keepdim=True)) / (task_summary.std(dim=-1, keepdim=True, unbiased=False) + 1e-6)
        if self.detach_encoder:
            task_summary = task_summary.detach()
        hidden = self.encoder(task_summary)
        mu = self.mu(hidden)
        logvar = self.logvar(hidden).clamp(-10.0, 10.0)
        if deterministic:
            u_tau = mu
        else:
            std = torch.exp(0.5 * logvar)
            u_tau = mu + torch.randn_like(std) * std
        meta_ib_loss = 0.5 * torch.mean(torch.sum(mu.pow(2) + logvar.exp() - logvar - 1.0, dim=-1))
        return mu, logvar, u_tau, meta_ib_loss


class PatchFSL(nn.Module):
    """
    Full PatchFSL Model
    """
    def __init__(self, data_args, model_args, task_args, PatchFSL_cfg, model='GWN'):
        # PatchFSL_cfg :{data_list, s im, K, patch_encoder, base_dir}
        super(PatchFSL,self).__init__()

        self.data_args, self.model_args, self.task_args, self.PatchFSL_cfg = data_args, model_args, task_args, PatchFSL_cfg
        self.use_pattern_ib = as_bool(PatchFSL_cfg.get('use_pattern_ib', False))
        self.use_meta_ib = as_bool(PatchFSL_cfg.get('use_meta_ib', False))
        self.meta_ib_use_label = as_bool(PatchFSL_cfg.get('meta_ib_use_label', True))
        self.meta_ib_modulate_pattern_query = as_bool(PatchFSL_cfg.get('meta_ib_modulate_pattern_query', True))
        self.meta_ib_dim = int(PatchFSL_cfg.get('meta_ib_dim', 32))
        self.use_eagt = as_bool(PatchFSL_cfg.get('use_eagt', False))
        self.eagt_debug = as_bool(PatchFSL_cfg.get('eagt_debug', False))
        self.eagt_sparse_loss_weight = float(PatchFSL_cfg.get('eagt_sparse_loss_weight', 0.0))
        self.eagt_evidence_loss_weight = float(PatchFSL_cfg.get('eagt_evidence_loss_weight', 0.0))
        self.use_sagt = as_bool(PatchFSL_cfg.get('use_sagt', False))
        self.sagt_debug = as_bool(PatchFSL_cfg.get('sagt_debug', False))
        self.use_crct = as_bool(PatchFSL_cfg.get('use_crct', False))
        self.crct_debug = as_bool(PatchFSL_cfg.get('crct_debug', False))
        self.latest_crct_debug = None
        self.latest_crct_aux_loss = {}
        self.latest_sagt_debug = None
        self.latest_sagt_aux_loss = {}
        self.latest_eagt_debug = None
        self._printed_online_eagt = False
        if self.use_sagt and self.use_crct:
            print("[SAGT] Both use_sagt and use_crct are enabled. In v1, SAGT is applied and CRCT is skipped.")
        if self.use_eagt and self.use_crct:
            print("[CRCT] Both use_eagt and use_crct are enabled. In v1, CRCT is applied after original TPB graph and EAGT is skipped.")
        # Pattern_Encoder
        
        pattern = torch.load(PatchFSL_cfg['base_dir'] / 'pattern/{}/{}_{}_cl.pt'.format(PatchFSL_cfg["data_list"],PatchFSL_cfg["sim"],PatchFSL_cfg["K"]))
        pattern = pattern.detach().to(self.PatchFSL_cfg['device'])
        
        Pattern_Encoder = PatternEncoder_patternkeyv2(
            pattern,
            use_pattern_ib=self.use_pattern_ib,
            pattern_ib_prior=PatchFSL_cfg.get('pattern_ib_prior', 0.05),
            pattern_ib_temperature=PatchFSL_cfg.get('pattern_ib_temperature', 1.0),
            pattern_ib_gate=PatchFSL_cfg.get('pattern_ib_gate', "gumbel_sigmoid"),
            pattern_ib_topk=PatchFSL_cfg.get('pattern_ib_topk', 0),
            pattern_ib_eps=PatchFSL_cfg.get('pattern_ib_eps', 1e-8),
            meta_ib_dim=self.meta_ib_dim,
            meta_ib_modulate_pattern_query=self.use_meta_ib and self.meta_ib_modulate_pattern_query
        )
        if self.use_meta_ib:
            meta_input_dim = 4 if self.meta_ib_use_label else 2
            self.task_ib_encoder = TaskIBEncoder(
                meta_input_dim,
                meta_ib_dim=self.meta_ib_dim,
                hidden_dim=max(32, self.meta_ib_dim * 2),
                detach_encoder=PatchFSL_cfg.get('meta_ib_detach_encoder', True)
            )
        else:
            self.task_ib_encoder = None
        if self.use_sagt:
            self.source_structure_cache = PatchFSL_cfg.get('source_structure_cache', None)
            if self.source_structure_cache is None:
                raise ValueError("use_sagt=True requires PatchFSL_cfg['source_structure_cache'].")
            self.source_structure_cache.to(self.PatchFSL_cfg['device'])
            self.sagt_graph_constructor = SAGTGraphConstructor(PatchFSL_cfg)
        else:
            self.source_structure_cache = None
            self.sagt_graph_constructor = None
        if self.use_crct:
            self.crct_graph_constructor = CRCTGraphConstructor(PatchFSL_cfg)
        else:
            self.crct_graph_constructor = None
        if self.use_eagt or self.use_sagt:
            self.source_evidence_cache = PatchFSL_cfg.get('source_evidence_cache', None)
            if self.source_evidence_cache is not None:
                self.source_evidence_cache.to(self.PatchFSL_cfg['device'])
            self.eagt_graph_constructor = EAGTGraphConstructor(PatchFSL_cfg) if self.use_eagt else None
        else:
            self.source_evidence_cache = None
            self.eagt_graph_constructor = None

        # FC_model
        model_count={'patch':2, 'pattern':2}
        num_model = model_count[PatchFSL_cfg['patch_encoder']]
        FCmodel = FCNet(num_model * model_args['mae']['out_channel'], model_args['mae']['out_channel'], task_args['maml']['pred_num'])

        # ST_model
        if(model == "GWN"):
            STmodel = BatchA_patch_gwnet(out_dim = model_args['mae']['out_channel'],supports_len = 2)
        else:
            raise NotImplementedError
        # Patch_model
        Reconsmodel = ReconstrucAdjNet(model_args['mae']['out_channel'])

        EncoderLayer = nn.TransformerEncoderLayer(d_model = 128, nhead=2, dim_feedforward = 128)
        Pattern_Day = nn.TransformerEncoder(EncoderLayer, num_layers = 1)
        
        # model_list
        self.model_list = nn.ModuleList([Pattern_Day, Pattern_Encoder, STmodel, FCmodel, Reconsmodel])

        print("[INFO]Pattern_Day has {} params, STmodel has {} params, FCmodel has {} params, Reconsmodel has {} params, Pattern_Encoder has {} params".format(count_parameters(Pattern_Day),count_parameters(STmodel), count_parameters(FCmodel),count_parameters(Reconsmodel), count_parameters(Pattern_Encoder)))

    
    def encode_task_ib(self, data_i, deterministic=False):
        if not self.use_meta_ib or self.task_ib_encoder is None:
            zero = torch.zeros((), device=self.PatchFSL_cfg['device'])
            return None, zero
        _, _, u_tau, meta_ib_loss = self.task_ib_encoder(
            data_i,
            self.PatchFSL_cfg['device'],
            use_label=self.meta_ib_use_label,
            deterministic=deterministic
        )
        return u_tau, meta_ib_loss

    def forward(self, data_i, A, stage = 'train', u_tau=None, return_ib_loss=False):
        Pattern_Day, Pattern_Encoder, STmodel, FCmodel, Reconsmodel = self.model_list
        
        if(stage == 'train'):
            Pattern_Encoder.train()
            Pattern_Day.train()
            FCmodel.train()
            STmodel.train()
            Reconsmodel.train()
            if self.sagt_graph_constructor is not None:
                self.sagt_graph_constructor.train()
        else:
            Pattern_Day.eval()
            Pattern_Encoder.eval()
            FCmodel.eval()
            STmodel.eval()
            Reconsmodel.eval()
            if self.sagt_graph_constructor is not None:
                self.sagt_graph_constructor.eval()
        x, y, means, stds = data_i.x, data_i.y, data_i.means, data_i.stds
        # print("x shape is : {}, y shape is : {}".format(x.shape, y.shape))
        x, y, means, stds, A = torch.tensor(x).to(self.PatchFSL_cfg['device']), torch.tensor(y).to(self.PatchFSL_cfg['device']),torch.tensor(means).to(self.PatchFSL_cfg['device']),torch.tensor(stds).to(self.PatchFSL_cfg['device']),torch.tensor(A,dtype=torch.float32).to(self.PatchFSL_cfg['device'])
        zero = torch.zeros((), device=x.device, dtype=x.dtype)
        ib_losses = {"pattern_ib_loss": zero, "meta_ib_loss": zero}
        ib_losses.update({"eagt_sparse_loss": zero, "eagt_evidence_loss": zero})
        ib_losses.update({
            "sagt_sparse_loss": zero,
            "sagt_rank_loss": zero,
            "sagt_res_loss": zero,
            "sagt_spec_loss": zero,
        })
        ib_losses.update({
            "crct_sparse_loss": zero,
            "crct_sharp_loss": zero,
            "crct_balance_loss": zero,
            "crct_consistency_loss": zero,
            "crct_relation_kd_loss": zero,
            "crct_unknown_reg_loss": zero,
        })
        # remember that the input of TSFormer is [B, N, 2, L]
        x = x.permute(0,1,3,2)
        # shape : [B, N, 12, 1]
        raw_x = x[:,:,0:1, -12:].permute(0,1,3,2)
        raw_x_1day = x[:,:,0:1, -288:].permute(0,1,3,2)
        eagt_history = raw_x_1day
        
        B, N, ___, __ = raw_x_1day.shape
        raw_x_1day = raw_x_1day.reshape(B, N, 24, 12)
        # raw_feature shape : [B, N, D]
        

        if(self.PatchFSL_cfg['patch_encoder'] == 'pattern'):
            # H : [B, N, 24, 12]
            H = raw_x_1day
            # day1pattern : [B, N, 24, D]
            if return_ib_loss:
                day1pattern, pattern_ib_loss = Pattern_Encoder(
                    H,
                    return_ib_loss=True,
                    training_ib=(stage == 'train'),
                    u_tau=u_tau
                )
                ib_losses["pattern_ib_loss"] = pattern_ib_loss
            else:
                day1pattern = Pattern_Encoder(H, u_tau=u_tau)
            BB, NN, LL, DD = day1pattern.shape
            # day1pattern : [L, BN, D]
            day1pattern = day1pattern.reshape(BB * NN, LL, DD).permute(1,0,2)
            # pattern : [L, BN, D]
            pattern = Pattern_Day(day1pattern, mask=None)
            # pattern : [B, N, D]
            pattern = pattern[-1:, :, :].squeeze(0).reshape(BB, NN, DD)
            
            
            A_patch = Reconsmodel(pattern)
            A_original = A_patch
            if self.use_sagt:
                A_patch, sagt_aux_loss, sagt_debug = self.sagt_graph_constructor(
                    eagt_history,
                    A_original=A_original,
                    source_structure_cache=self.source_structure_cache,
                    source_evidence_cache=self.source_evidence_cache,
                    return_debug=self.sagt_debug
                )
                for key, value in sagt_aux_loss.items():
                    ib_losses[key] = value
                self.latest_sagt_aux_loss = {
                    key: value.detach() for key, value in sagt_aux_loss.items()
                }
                self.latest_sagt_debug = sagt_debug if self.sagt_debug else None
                self.latest_crct_debug = None
                self.latest_eagt_debug = None
            elif self.use_crct:
                A_patch, crct_aux_loss, crct_debug = self.crct_graph_constructor(
                    eagt_history,
                    A_original=A_original,
                    return_debug=self.crct_debug
                )
                for key, value in crct_aux_loss.items():
                    ib_losses[key] = value
                self.latest_crct_aux_loss = {
                    key: value.detach() for key, value in crct_aux_loss.items()
                }
                self.latest_crct_debug = crct_debug if self.crct_debug else None
                self.latest_eagt_debug = None
            elif self.use_eagt:
                source_cache = self.source_evidence_cache
                if source_cache is None:
                    if not self._printed_online_eagt:
                        print("[EAGT] using online mini-batch source evidence cache")
                        self._printed_online_eagt = True
                    source_cache = SourceEvidenceCache(device=self.PatchFSL_cfg['device']).build_from_source_data(
                        {"online": eagt_history.detach().cpu()},
                        args=self.PatchFSL_cfg
                    ).to(self.PatchFSL_cfg['device'])
                A_patch, eagt_aux_loss, eagt_debug = self.eagt_graph_constructor(
                    eagt_history,
                    A_original=A_patch,
                    source_cache=source_cache,
                    return_debug=self.eagt_debug
                )
                ib_losses["eagt_sparse_loss"] = eagt_aux_loss["eagt_sparse_loss"]
                ib_losses["eagt_evidence_loss"] = eagt_aux_loss["eagt_evidence_loss"]
                self.latest_eagt_debug = eagt_debug if self.eagt_debug else None
                self.latest_crct_debug = None
                self.latest_sagt_debug = None
            else:
                self.latest_eagt_debug = None
                self.latest_crct_debug = None
                self.latest_sagt_debug = None
        
        A_list = [A_patch, A_patch.permute(0,2,1)]
        
        raw_emb, Ax = STmodel(raw_x,A_list)

        
        

        if(self.PatchFSL_cfg['patch_encoder'] == 'raw'):
            input_features = [raw_emb]
        elif(self.PatchFSL_cfg['patch_encoder'] == 'pattern'):
            input_features = [raw_emb, pattern]
        input_features = torch.cat(input_features, dim = 2)

        out = FCmodel(input_features)
        
        # unnorm
        out = unnorm(out, means, stds)
        
        if return_ib_loss:
            return out,y, Ax, ib_losses
        return out,y, Ax

class STRep(nn.Module):
    """
    Reptile-based Few-shot learning architecture for STGNN
    """
    def __init__(self, data_args, task_args, model_args,PatchFSL_cfg, model='GWN'):
        super(STRep, self).__init__()
        self.data_args = data_args
        self.task_args = task_args
        self.model_args = model_args
        self.PatchFSL_cfg = PatchFSL_cfg

        self.update_lr = model_args['STnet']['update_lr']
        # self.update_lr = 0.0005
        self.meta_lr = model_args['STnet']['meta_lr']
        self.update_step = model_args['STnet']['update_step']
        # update_step_test is not used. It is replaced by target_epochs in main.
        self.task_num = task_args['maml']['task_num']
        self.model_name = model
        self.device = PatchFSL_cfg['device']
        self.current_epoch = 0
        self.use_pattern_ib = as_bool(PatchFSL_cfg.get('use_pattern_ib', False))
        self.use_meta_ib = as_bool(PatchFSL_cfg.get('use_meta_ib', False))
        self.pattern_ib_weight = float(PatchFSL_cfg.get('pattern_ib_weight', 0.0))
        self.meta_ib_weight = float(PatchFSL_cfg.get('meta_ib_weight', 0.0))
        self.use_eagt = as_bool(PatchFSL_cfg.get('use_eagt', False))
        self.eagt_sparse_loss_weight = float(PatchFSL_cfg.get('eagt_sparse_loss_weight', 0.0))
        self.eagt_evidence_loss_weight = float(PatchFSL_cfg.get('eagt_evidence_loss_weight', 0.0))
        self.use_sagt = as_bool(PatchFSL_cfg.get('use_sagt', False))
        self.sagt_sparse_loss_weight = float(PatchFSL_cfg.get('sagt_sparse_loss_weight', 0.0))
        self.sagt_rank_loss_weight = float(PatchFSL_cfg.get('sagt_rank_loss_weight', 0.0))
        self.sagt_res_loss_weight = float(PatchFSL_cfg.get('sagt_res_loss_weight', 0.0))
        self.sagt_spec_loss_weight = float(PatchFSL_cfg.get('sagt_spec_loss_weight', 0.0))
        self.use_crct = as_bool(PatchFSL_cfg.get('use_crct', False))
        self.crct_sparse_loss_weight = float(PatchFSL_cfg.get('crct_sparse_loss_weight', 0.0))
        self.crct_sharp_loss_weight = float(PatchFSL_cfg.get('crct_sharp_loss_weight', 0.0))
        self.crct_balance_loss_weight = float(PatchFSL_cfg.get('crct_balance_loss_weight', 0.0))
        self.crct_consistency_loss_weight = float(PatchFSL_cfg.get('crct_consistency_loss_weight', 0.0))
        self.crct_relation_kd_loss_weight = float(PatchFSL_cfg.get('crct_relation_kd_weight', PatchFSL_cfg.get('crct_relation_kd_loss_weight', 0.0)))
        self.crct_unknown_reg_loss_weight = float(PatchFSL_cfg.get('crct_unknown_reg_weight', PatchFSL_cfg.get('crct_unknown_reg_loss_weight', 0.0)))
        self.debug_max_batches = int(PatchFSL_cfg.get('debug_max_batches', -1))
        self.ib_enabled = self.use_pattern_ib or self.use_meta_ib
        self.extra_loss_enabled = self.ib_enabled or self.use_eagt or self.use_crct or self.use_sagt
        self.last_ib_log = {}
        self.last_crct_log = {}
        self.last_sagt_log = {}

        self.model = PatchFSL(data_args, model_args, task_args,PatchFSL_cfg).to(self.device)
        for name, params in self.model.named_parameters():
            print("{} : {}, require_grads : {}".format(name, params.shape,params.requires_grad))
            
        # print(self.model)
        print("model params: ", count_parameters(self.model))

        self.meta_optim = optim.AdamW(self.model.parameters(), lr=self.meta_lr, weight_decay=1e-2)
        # self.meta_optim = torch.optim.SGD(self.model.parameters(), lr=self.update_lr, momentum=0.9)
        self.loss_criterion = nn.MSELoss(reduction='mean')

    def zero_loss(self):
        return torch.zeros((), device=self.device)

    def combine_losses(self, pred_loss, pattern_ib_loss=None, meta_ib_loss=None,
                       eagt_sparse_loss=None, eagt_evidence_loss=None,
                       crct_sparse_loss=None, crct_sharp_loss=None,
                       crct_balance_loss=None, crct_consistency_loss=None,
                       crct_relation_kd_loss=None, crct_unknown_reg_loss=None,
                       sagt_sparse_loss=None, sagt_rank_loss=None,
                       sagt_res_loss=None, sagt_spec_loss=None):
        total_loss = pred_loss
        if self.use_pattern_ib and pattern_ib_loss is not None:
            total_loss = total_loss + self.pattern_ib_weight * pattern_ib_loss
        if self.use_meta_ib and meta_ib_loss is not None:
            total_loss = total_loss + self.meta_ib_weight * meta_ib_loss
        if self.use_eagt and eagt_sparse_loss is not None:
            total_loss = total_loss + self.eagt_sparse_loss_weight * eagt_sparse_loss
        if self.use_eagt and eagt_evidence_loss is not None:
            total_loss = total_loss + self.eagt_evidence_loss_weight * eagt_evidence_loss
        if self.use_crct:
            if crct_sparse_loss is not None:
                total_loss = total_loss + self.crct_sparse_loss_weight * crct_sparse_loss
            if crct_sharp_loss is not None:
                total_loss = total_loss + self.crct_sharp_loss_weight * crct_sharp_loss
            if crct_balance_loss is not None:
                total_loss = total_loss + self.crct_balance_loss_weight * crct_balance_loss
            if crct_consistency_loss is not None:
                total_loss = total_loss + self.crct_consistency_loss_weight * crct_consistency_loss
            if crct_relation_kd_loss is not None:
                total_loss = total_loss + self.crct_relation_kd_loss_weight * crct_relation_kd_loss
            if crct_unknown_reg_loss is not None:
                total_loss = total_loss + self.crct_unknown_reg_loss_weight * crct_unknown_reg_loss
        if self.use_sagt:
            if sagt_sparse_loss is not None:
                total_loss = total_loss + self.sagt_sparse_loss_weight * sagt_sparse_loss
            if sagt_rank_loss is not None:
                total_loss = total_loss + self.sagt_rank_loss_weight * sagt_rank_loss
            if sagt_res_loss is not None:
                total_loss = total_loss + self.sagt_res_loss_weight * sagt_res_loss
            if sagt_spec_loss is not None:
                total_loss = total_loss + self.sagt_spec_loss_weight * sagt_spec_loss
        return total_loss

    def _crct_loss_args(self, ib_losses):
        return [
            ib_losses.get("crct_sparse_loss", None),
            ib_losses.get("crct_sharp_loss", None),
            ib_losses.get("crct_balance_loss", None),
            ib_losses.get("crct_consistency_loss", None),
            ib_losses.get("crct_relation_kd_loss", None),
            ib_losses.get("crct_unknown_reg_loss", None),
        ]

    def _append_crct_logs(self, store, ib_losses, total_loss):
        if not self.use_crct:
            return
        for key in [
            "crct_sparse_loss", "crct_sharp_loss", "crct_balance_loss",
            "crct_consistency_loss", "crct_relation_kd_loss",
            "crct_unknown_reg_loss",
        ]:
            value = ib_losses.get(key, None)
            if value is not None:
                store.setdefault(key, []).append(value.detach().item())
        store.setdefault("total_loss", []).append(total_loss.detach().item())

    def _sagt_loss_args(self, ib_losses):
        return [
            ib_losses.get("sagt_sparse_loss", None),
            ib_losses.get("sagt_rank_loss", None),
            ib_losses.get("sagt_res_loss", None),
            ib_losses.get("sagt_spec_loss", None),
        ]

    def _append_sagt_logs(self, store, ib_losses, total_loss):
        if not self.use_sagt:
            return
        for key in ["sagt_sparse_loss", "sagt_rank_loss", "sagt_res_loss", "sagt_spec_loss"]:
            value = ib_losses.get(key, None)
            if value is not None:
                store.setdefault(key, []).append(value.detach().item())
        store.setdefault("total_loss", []).append(total_loss.detach().item())

    def _predict_loss(self, out, y, meta_graph, matrix, stage='source'):
        if self.model_name in ['v_GRU', 'r_GRU', 'v_STGCN']:
            return self.loss_criterion(out, y)
        return self.calculate_loss(out, y, meta_graph, matrix, stage, graph_loss=False)

    def _forward_with_optional_ib(self, data_i, A_gnd, stage='train', u_tau=None):
        if self.extra_loss_enabled:
            out, y, meta_graph, ib_losses = self.model(data_i, A_gnd, stage=stage, u_tau=u_tau, return_ib_loss=True)
        else:
            out, y, meta_graph = self.model(data_i, A_gnd, stage=stage)
            zero = self.zero_loss()
            ib_losses = {"pattern_ib_loss": zero, "meta_ib_loss": zero}
        return out, y, meta_graph, ib_losses

    def _encode_task_ib(self, data_i, deterministic=False):
        if not self.use_meta_ib:
            return None, self.zero_loss()
        return self.model.encode_task_ib(data_i, deterministic=deterministic)

    def checkpoint_state(self, epoch, optimizer=None, config=None, best_metric=None, stage="meta"):
        return {
            "stage": stage,
            "epoch": epoch,
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "config": config,
            "best_metric": best_metric,
            "use_pattern_ib": self.use_pattern_ib,
            "use_meta_ib": self.use_meta_ib,
            "use_eagt": self.use_eagt,
            "use_crct": self.use_crct,
            "use_sagt": self.use_sagt
        }
    
    def get_per_step_loss_importance_vector(self):
        """
        Generates a tensor of dimensionality (update_step) indicating the importance of each step's target
        loss towards the optimization loss.
        :return: A tensor to be used to compute the weighted average of the loss, useful for
        the MSL (Multi Step Loss) mechanism.
        """
        loss_weights = np.ones(shape=(self.update_step)) * (
                1.0 / self.update_step)
        decay_rate = 1.0 / self.update_step / self.task_args['maml']['train_epochs']
        # print("decay_rate : {}".format(decay_rate))
        min_value_for_non_final_losses = 0.03 / self.update_step
        for i in range(len(loss_weights) - 1):
            curr_value = np.maximum(loss_weights[i] - (self.current_epoch * decay_rate), min_value_for_non_final_losses)
            # print("each step : {}, {}".format(self.current_epoch * decay_rate, min_value_for_non_final_losses))
            loss_weights[i] = curr_value
        # print("loss_weights : {}".format(loss_weights))

        curr_value = np.minimum(
            loss_weights[-1] + (self.current_epoch * (self.update_step - 1) * decay_rate),
            1.0 - ((self.update_step - 1) * min_value_for_non_final_losses))
        loss_weights[-1] = curr_value
        loss_weights = torch.Tensor(loss_weights).to(device=self.device)
        return loss_weights
    
    def graph_reconstruction_loss(self, meta_graph, adj_graph):
        adj_graph = adj_graph.unsqueeze(0).float()
        for i in range(meta_graph.shape[0]):
            if i == 0:
                matrix = adj_graph
            else:
                matrix = torch.cat((matrix, adj_graph), 0)
        criteria = nn.MSELoss()
        loss = criteria(meta_graph, matrix.float())
        return loss
    
    def calculate_loss(self, out, y, meta_graph, matrix, stage='target', graph_loss=True, loss_lambda=0):
        if loss_lambda == 0:
            loss = self.loss_criterion(out, y)
        if graph_loss:
            if stage == 'source' or stage == 'target_maml':
                loss_predict = self.loss_criterion(out, y)
                loss_reconsturct = self.graph_reconstruction_loss(meta_graph, matrix)
            else:
                loss_predict = self.loss_criterion(out, y)
                loss_reconsturct = self.loss_criterion(meta_graph, matrix.float())
            loss = loss_predict + loss_lambda * loss_reconsturct
        else:
            loss = self.loss_criterion(out, y)

        return loss
    
    
    
    def meta_train_revise(self, data_spt, matrix_spt, data_qry, matrix_qry):
        loss_weights = self.get_per_step_loss_importance_vector().detach()
        # init_model = deepcopy(self.model)
        init_params = deepcopy(list(self.model.parameters()))
        total_mae = []
        total_mse = []
        total_rmse = []
        total_mape = []
        total_pred_loss = []
        total_pattern_ib_loss = []
        total_meta_ib_loss = []
        total_total_loss = []
        total_crct_loss = {}
        total_sagt_loss = {}
        
        # taskwise loss, precision
        task_losses = []
        active_task_num = self.task_num
        if self.debug_max_batches > 0:
            active_task_num = min(active_task_num, self.debug_max_batches, len(data_spt), len(data_qry))
        for i in range(active_task_num):
            # maml_params = deepcopy(init_params)
            for idx, (init_param, model_param) in enumerate(zip(init_params, self.model.parameters())):
                model_param.data = init_param
            task_loss = 0
            for k in range(self.update_step):
                batch_size, node_num, seq_len, _ = data_spt[i].x.shape
                if self.model_name == 'GWN':
                    A = matrix_spt[i]
                    A = A.unsqueeze(0).float()
                    for batch_i in range(batch_size):
                        if batch_i == 0:
                            A_gnd = A
                        else:
                            A_gnd = torch.cat((A_gnd, A), 0)
                    A_gnd = torch.tensor(A_gnd,dtype=torch.float32).to(self.device)
                else:
                    A_gnd = matrix_spt[i].to(self.device)

                # 1: use maml para to compute output
                # for idx, (maml_param, model_param) in enumerate(zip(maml_params, self.model.parameters())):
                #     model_param.data = maml_param
                u_tau, meta_ib_loss = self._encode_task_ib(data_spt[i], deterministic=False)
                out, y, meta_graph, ib_losses = self._forward_with_optional_ib(data_spt[i], A_gnd, u_tau=u_tau)
                
                # 2: use the output to compute the grad
                pred_loss = self._predict_loss(out, y, meta_graph, matrix_spt[i], 'source')
                loss = self.combine_losses(
                    pred_loss,
                    ib_losses["pattern_ib_loss"],
                    meta_ib_loss,
                    ib_losses.get("eagt_sparse_loss", None),
                    ib_losses.get("eagt_evidence_loss", None),
                    *self._crct_loss_args(ib_losses),
                    *self._sagt_loss_args(ib_losses)
                )
                grad = torch.autograd.grad(loss, self.model.parameters(), allow_unused=True, retain_graph=self.extra_loss_enabled)
                grad = list(grad)
                for idx,(a, model_param) in enumerate(zip(grad, self.model.parameters())):
                    if(a==None):
                        grad[idx] = torch.zeros_like(model_param)
                
                # 3 : use the grad to update maml parameters
                # maml_params = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, maml_params)))
                for idx, (gra, model_param) in enumerate(zip(grad, self.model.parameters())):
                    model_param.data = model_param.data - self.update_lr * gra
                    
                del grad
                # Then, calculate the loss of query task.
                batch_size, node_num, seq_len, _ = data_qry[i].x.shape
                if self.model_name == 'GWN':
                    A = matrix_qry[i]
                    A = A.unsqueeze(0).float()
                    for batch_i in range(batch_size):
                        if batch_i == 0:
                            A_gnd = A
                        else:
                            A_gnd = torch.cat((A_gnd, A), 0)
                    A_gnd = torch.tensor(A_gnd,dtype=torch.float32).to(self.device)
                else:
                    A_gnd = matrix_qry[i].to(self.device)
                    
                # 1. use the parameter after update to compute output
                # for idx, (maml_param, model_param) in enumerate(zip(maml_params, self.model.parameters())):
                #     model_param.data = maml_param
                out, y, meta_graph, ib_losses_q = self._forward_with_optional_ib(data_qry[i], A_gnd, u_tau=u_tau)
                
                
                # 2: use the output to compute the grad
                pred_loss_q = self._predict_loss(out, y, meta_graph, matrix_spt[i], 'source')
                loss_q = self.combine_losses(
                    pred_loss_q,
                    ib_losses_q["pattern_ib_loss"],
                    meta_ib_loss,
                    ib_losses_q.get("eagt_sparse_loss", None),
                    ib_losses_q.get("eagt_evidence_loss", None),
                    *self._crct_loss_args(ib_losses_q),
                    *self._sagt_loss_args(ib_losses_q)
                )
                if self.ib_enabled:
                    total_pred_loss.append(pred_loss_q.detach().item())
                    total_pattern_ib_loss.append(ib_losses_q["pattern_ib_loss"].detach().item())
                    total_meta_ib_loss.append(meta_ib_loss.detach().item())
                    total_total_loss.append(loss_q.detach().item())
                self._append_crct_logs(total_crct_loss, ib_losses_q, loss_q)
                self._append_sagt_logs(total_sagt_loss, ib_losses_q, loss_q)
                
                # grad_q = torch.autograd.grad(loss, self.model.parameters(), allow_unused=True, retain_graph = True)
                # grad_q = list(grad_q)
                # for idx,a in enumerate(grad_q):
                #     if(a==None):
                #         grad_q[idx] = torch.tensor(0.0)

                # 3 : memorize the grad
                # grads_list[k] = loss_weights[k] * grad_q
                # then add the weighted loss to the task_loss
                task_loss += loss_weights[k] * loss_q
                del loss_q
                # print("{} :".format(k))
                # print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
                # print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
                # print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
                
            
                if(k == self.update_step - 1):
                    MSE,RMSE,MAE,MAPE = calc_metric(out, y)
                    total_mse.append(MSE.cpu().detach().numpy())
                    total_rmse.append(RMSE.cpu().detach().numpy())
                    total_mae.append(MAE.cpu().detach().numpy())
                    total_mape.append(MAPE.cpu().detach().numpy())
            
            
            task_losses.append(task_loss)
        # 2. use model loss to compute grads
        model_loss = torch.sum(torch.stack(task_losses))
        # # self.model.load_state_dict(init_model.state_dict())
        grad = torch.autograd.grad(model_loss, self.model.parameters(), allow_unused=True)
        grad = list(grad)
        for idx,(a, model_param) in enumerate(zip(grad, self.model.parameters())):
            if(a==None):
                grad[idx] = torch.zeros_like(model_param)
        for init_param, model_param, gra in zip(init_params, self.model.parameters(),grad):
            model_param.data = init_param - self.update_lr * gra
        # 3. use grads_list to do gradien descent
        # for init_param, model_param in zip(init_params, self.model.parameters()):
        #     model_param.data = init_param
        # for tsk in range(self.task_num):
        #     for k in range(self.update_step):
        #         maml_params = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grads_task_list[tsk][k], self.model.parameters())))
            
        # for idx, para in enumerate(self.model.parameters()):
        #     print("model para : {}, init para : {}".format(para, init_params[idx]))
        
        
        # self.meta_optim.zero_grad()
        # model_loss.backward()
        # self.meta_optim.step()
        self.current_epoch+=1
        if self.ib_enabled:
            self.last_ib_log = {
                "pred_loss": float(np.mean(total_pred_loss)) if len(total_pred_loss) > 0 else 0.0,
                "pattern_ib_loss": float(np.mean(total_pattern_ib_loss)) if len(total_pattern_ib_loss) > 0 else 0.0,
                "meta_ib_loss": float(np.mean(total_meta_ib_loss)) if len(total_meta_ib_loss) > 0 else 0.0,
                "total_loss": float(np.mean(total_total_loss)) if len(total_total_loss) > 0 else 0.0
            }
        if self.use_crct:
            self.last_crct_log = {
                key: float(np.mean(value)) if len(value) > 0 else 0.0
                for key, value in total_crct_loss.items()
            }
        if self.use_sagt:
            self.last_sagt_log = {
                key: float(np.mean(value)) if len(value) > 0 else 0.0
                for key, value in total_sagt_loss.items()
            }

        # return MSELoss, mse, rmse, mae, mape
        return model_loss.detach().cpu().numpy(),np.mean(total_mse), np.mean(total_rmse), np.mean(total_mae),np.mean(total_mape)

    def forward(self, data, matrix):
        out, meta_graph = self.model(data, matrix)
        return out, meta_graph

    def train_batch(self, start,end,source_dataset,loss_fn,opts):
        total_loss = []
        total_mae = []
        total_mse = []
        total_rmse = []
        total_mape = []
        total_pred_loss = []
        total_pattern_ib_loss = []
        total_meta_ib_loss = []
        total_crct_loss = {}
        total_sagt_loss = {}
        
        if self.debug_max_batches > 0:
            end = min(end, start + self.debug_max_batches)
        for idx in range(start,end):
            data_i, A = source_dataset[idx]
            B, N, D, L = data_i.x.shape
            A = A.unsqueeze(0).float()
            for i in range(B):
                if i == 0:
                    A_gnd = A
                else:
                    A_gnd = torch.cat((A_gnd, A), 0)
            A_gnd = torch.tensor(A_gnd,dtype=torch.float32).to(self.device)

            if self.extra_loss_enabled:
                u_tau, meta_ib_loss = self._encode_task_ib(data_i, deterministic=False)
                out,y,Ax,ib_losses = self.model(data_i, A_gnd, u_tau=u_tau, return_ib_loss=True)
                pred_loss = loss_fn(out, y)
                loss = self.combine_losses(
                    pred_loss,
                    ib_losses["pattern_ib_loss"],
                    meta_ib_loss,
                    ib_losses.get("eagt_sparse_loss", None),
                    ib_losses.get("eagt_evidence_loss", None),
                    *self._crct_loss_args(ib_losses),
                    *self._sagt_loss_args(ib_losses)
                )
                total_pred_loss.append(pred_loss.detach().item())
                total_pattern_ib_loss.append(ib_losses["pattern_ib_loss"].detach().item())
                total_meta_ib_loss.append(meta_ib_loss.detach().item())
                self._append_crct_logs(total_crct_loss, ib_losses, loss)
                self._append_sagt_logs(total_sagt_loss, ib_losses, loss)
            else:
                out,y,Ax = self.model(data_i, A_gnd)
                loss = loss_fn(out, y)

            for opt in opts:
                opt.zero_grad()
            loss.backward()
            for opt in opts:
                opt.step()
        
            MSE,RMSE,MAE,MAPE = calc_metric(out, y)
            
            total_mse.append(MSE.cpu().detach().numpy())
            total_rmse.append(RMSE.cpu().detach().numpy())
            total_mae.append(MAE.cpu().detach().numpy())
            total_mape.append(MAPE.cpu().detach().numpy())
            total_loss.append(loss.item())
        if self.ib_enabled:
            self.last_ib_log = {
                "pred_loss": float(np.mean(total_pred_loss)) if len(total_pred_loss) > 0 else 0.0,
                "pattern_ib_loss": float(np.mean(total_pattern_ib_loss)) if len(total_pattern_ib_loss) > 0 else 0.0,
                "meta_ib_loss": float(np.mean(total_meta_ib_loss)) if len(total_meta_ib_loss) > 0 else 0.0,
                "total_loss": float(np.mean(total_loss)) if len(total_loss) > 0 else 0.0
            }
        if self.use_crct:
            self.last_crct_log = {
                key: float(np.mean(value)) if len(value) > 0 else 0.0
                for key, value in total_crct_loss.items()
            }
        if self.use_sagt:
            self.last_sagt_log = {
                key: float(np.mean(value)) if len(value) > 0 else 0.0
                for key, value in total_sagt_loss.items()
            }
        return total_mse,total_rmse, total_mae, total_mape, total_loss

    def test_batch(self,start,end,source_dataset,stage = "test"):
        total_loss = []
        total_mae = []
        total_mse = []
        total_rmse = []
        total_mape = []
        
        with torch.no_grad():
            if self.debug_max_batches > 0:
                end = min(end, start + self.debug_max_batches)
            for idx in range(start,end):
                data_i, A = source_dataset[idx]
                B, N, D, L = data_i.x.shape
                A = A.unsqueeze(0).float()
                for i in range(B):
                    if i == 0:
                        A_gnd = A
                    else:
                        A_gnd = torch.cat((A_gnd, A), 0)
                A_gnd = torch.tensor(A_gnd,dtype=torch.float32).to(self.device)
                # A_gnd = A
                
                # activate .eval()
                if self.extra_loss_enabled:
                    u_tau, _ = self._encode_task_ib(data_i, deterministic=True)
                    out,y,Ax,ib_losses = self.model(data_i, A_gnd, stage='test', u_tau=u_tau, return_ib_loss=True)
                else:
                    out,y,Ax = self.model(data_i, A_gnd, stage='test')
                
                # print 12 horizons
                MSE,RMSE,MAE,MAPE = calc_metric(out, y, stage='test')
                total_mse.append(MSE.cpu().detach().numpy())
                total_rmse.append(RMSE.cpu().detach().numpy())
                total_mae.append(MAE.cpu().detach().numpy())
                total_mape.append(MAPE.cpu().detach().numpy())
        return total_mse,total_rmse, total_mae, total_mape, total_loss


    def finetuning(self, finetune_dataset, test_dataset, target_epochs, resume_path=None):
        """
        finetunning stage in MAML
        """
        if target_epochs <= 0:
            print("[INFO] Skip finetune phase because target_epochs <= 0")
            return
        curr_time = datetime.datetime.now().strftime(
    "%Y%m%d-%H%M%S")
        checkpoint_enabled = as_bool(self.PatchFSL_cfg.get('enable_checkpoint', True)) and (self.ib_enabled or self.use_eagt or self.use_crct or self.use_sagt)
        if checkpoint_enabled:
            model_path = Path(self.PatchFSL_cfg.get('checkpoint_dir', self.PatchFSL_cfg.get('ib_save_dir', './save/ib_runs')))
            checkpoint_prefix = self.PatchFSL_cfg.get('checkpoint_prefix', 'tpb_ib')
            target_city = self.PatchFSL_cfg.get('test_dataset', 'target')
            save_every = max(1, int(self.PatchFSL_cfg.get('save_every', 1)))
            save_best_only = bool(self.PatchFSL_cfg.get('save_best_only', False))
            overwrite_checkpoint = bool(self.PatchFSL_cfg.get('overwrite_checkpoint', False))
        else:
            model_path = Path('./save/meta_model/{}/{}/'.format(self.PatchFSL_cfg['data_list'],curr_time))
        
        if(not os.path.exists(model_path)):
            os.makedirs(model_path)
        print("Finetuned model saved in {}".format(model_path))
        optimizer = optim.Adam(self.model.parameters(), lr=self.meta_lr, weight_decay=1e-2)

        best_loss = 9999999999999.0
        best_model = None
        start_epoch = 0
        if resume_path:
            start_epoch, loaded_best = load_checkpoint(resume_path, self, optimizer)
            if loaded_best is not None:
                best_loss = loaded_best
            best_model = copy.deepcopy(self.model)
            print("INFO: Finetune resume loaded: {}, start_epoch={}".format(resume_path, start_epoch))
        print("[INFO] Enter finetune phase")

        for i in range(start_epoch, target_epochs):
            length = finetune_dataset.__len__()
            # length=40
            print('----------------------')
            time_1 = time.time()

            total_mse,total_rmse, total_mae, total_mape, total_loss = self.train_batch(0,length, finetune_dataset,self.loss_criterion,[optimizer])
            print('Epochs {}/{}'.format(i,target_epochs))
            print('in training   Unnormed MSE : {:.5f}, RMSE : {:.5f}, MAE : {:.5f}, MAPE: {:.5f}, normed MSE : {:.5f}.'.format(np.mean(total_mse), np.mean(total_rmse), np.mean(total_mae),np.mean(total_mape),np.mean(total_loss)))
            if self.ib_enabled:
                print("IB loss pred={:.5f}, pattern={:.5f}, meta={:.5f}, total={:.5f}".format(
                    self.last_ib_log.get("pred_loss", 0.0),
                    self.last_ib_log.get("pattern_ib_loss", 0.0),
                    self.last_ib_log.get("meta_ib_loss", 0.0),
                    self.last_ib_log.get("total_loss", 0.0)
                ))
            if self.use_crct:
                print("CRCT loss sparse={:.5f}, sharp={:.5f}, balance={:.5f}, kd={:.5f}, unknown={:.5f}, total={:.5f}".format(
                    self.last_crct_log.get("crct_sparse_loss", 0.0),
                    self.last_crct_log.get("crct_sharp_loss", 0.0),
                    self.last_crct_log.get("crct_balance_loss", 0.0),
                    self.last_crct_log.get("crct_relation_kd_loss", 0.0),
                    self.last_crct_log.get("crct_unknown_reg_loss", 0.0),
                    self.last_crct_log.get("total_loss", 0.0)
                ))
            if self.use_sagt:
                print("SAGT loss sparse={:.5f}, rank={:.5f}, res={:.5f}, spec={:.5f}, total={:.5f}".format(
                    self.last_sagt_log.get("sagt_sparse_loss", 0.0),
                    self.last_sagt_log.get("sagt_rank_loss", 0.0),
                    self.last_sagt_log.get("sagt_res_loss", 0.0),
                    self.last_sagt_log.get("sagt_spec_loss", 0.0),
                    self.last_sagt_log.get("total_loss", 0.0)
                ))
            
            mae_loss = np.mean(total_mae)
            if(mae_loss < best_loss):
                best_model = copy.deepcopy(self.model)
                best_loss = mae_loss
                if checkpoint_enabled:
                    save_checkpoint(
                        model_path / '{}_{}_best.pt'.format(checkpoint_prefix, target_city),
                        self.checkpoint_state(i, optimizer, self.PatchFSL_cfg.get('config', None), best_loss, stage="finetune"),
                        overwrite=overwrite_checkpoint
                    )
                else:
                    torch.save(self.model.state_dict(), model_path / 'finetuned_bestmodel.pt')
                print('Best model. Saved.')
            if checkpoint_enabled and (not save_best_only) and ((i + 1) % save_every == 0):
                state = self.checkpoint_state(i, optimizer, self.PatchFSL_cfg.get('config', None), best_loss, stage="finetune")
                save_checkpoint(model_path / '{}_{}_last.pt'.format(checkpoint_prefix, target_city), state, overwrite=overwrite_checkpoint)
                save_checkpoint(model_path / '{}_{}_finetune_epoch{}.pt'.format(checkpoint_prefix, target_city, i), state, overwrite=overwrite_checkpoint)
            print('this epoch costs {:.5}s'.format(time.time()-time_1))

        print("[INFO] Enter test phase")
        length = test_dataset.__len__()
        if best_model is None:
            best_model = copy.deepcopy(self.model)
        self.model = copy.deepcopy(best_model)
        
        total_mse_horizon,total_rmse_horizon, total_mae_horizon, total_mape_horizon, total_loss = self.test_batch(0,length, test_dataset,"test")
        for i in range(self.task_args['maml']['pred_num']):
            total_mae = []
            total_mse = []
            total_rmse = []
            total_mape = []
            for j in range(len(total_mse_horizon)):
                
                total_mse.append(total_mse_horizon[j][i])
                total_rmse.append(total_rmse_horizon[j][i])
                total_mae.append(total_mae_horizon[j][i])
                total_mape.append(total_mape_horizon[j][i])
                
            print('Horizon {} : Unnormed MSE : {:.5f}, RMSE : {:.5f}, MAE : {:.5f}, MAPE: {:.5f}'.format(i,np.mean(total_mse), np.mean(total_rmse), np.mean(total_mae),np.mean(total_mape)))

        
