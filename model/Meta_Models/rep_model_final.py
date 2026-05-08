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

from meta_patch import *
from reconstruction import *
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
        self.use_pattern_ib = bool(PatchFSL_cfg.get('use_pattern_ib', False))
        self.use_meta_ib = bool(PatchFSL_cfg.get('use_meta_ib', False))
        self.meta_ib_use_label = bool(PatchFSL_cfg.get('meta_ib_use_label', True))
        self.meta_ib_modulate_pattern_query = bool(PatchFSL_cfg.get('meta_ib_modulate_pattern_query', True))
        self.meta_ib_dim = int(PatchFSL_cfg.get('meta_ib_dim', 32))
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
        else:
            Pattern_Day.eval()
            Pattern_Encoder.eval()
            FCmodel.eval()
            STmodel.eval()
            Reconsmodel.eval()
        x, y, means, stds = data_i.x, data_i.y, data_i.means, data_i.stds
        # print("x shape is : {}, y shape is : {}".format(x.shape, y.shape))
        x, y, means, stds, A = torch.tensor(x).to(self.PatchFSL_cfg['device']), torch.tensor(y).to(self.PatchFSL_cfg['device']),torch.tensor(means).to(self.PatchFSL_cfg['device']),torch.tensor(stds).to(self.PatchFSL_cfg['device']),torch.tensor(A,dtype=torch.float32).to(self.PatchFSL_cfg['device'])
        zero = torch.zeros((), device=x.device, dtype=x.dtype)
        ib_losses = {"pattern_ib_loss": zero, "meta_ib_loss": zero}
        # remember that the input of TSFormer is [B, N, 2, L]
        x = x.permute(0,1,3,2)
        # shape : [B, N, 12, 1]
        raw_x = x[:,:,0:1, -12:].permute(0,1,3,2)
        raw_x_1day = x[:,:,0:1, -288:].permute(0,1,3,2)
        
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
        self.use_pattern_ib = bool(PatchFSL_cfg.get('use_pattern_ib', False))
        self.use_meta_ib = bool(PatchFSL_cfg.get('use_meta_ib', False))
        self.pattern_ib_weight = float(PatchFSL_cfg.get('pattern_ib_weight', 0.0))
        self.meta_ib_weight = float(PatchFSL_cfg.get('meta_ib_weight', 0.0))
        self.ib_enabled = self.use_pattern_ib or self.use_meta_ib
        self.last_ib_log = {}

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

    def combine_losses(self, pred_loss, pattern_ib_loss=None, meta_ib_loss=None):
        total_loss = pred_loss
        if self.use_pattern_ib and pattern_ib_loss is not None:
            total_loss = total_loss + self.pattern_ib_weight * pattern_ib_loss
        if self.use_meta_ib and meta_ib_loss is not None:
            total_loss = total_loss + self.meta_ib_weight * meta_ib_loss
        return total_loss

    def _predict_loss(self, out, y, meta_graph, matrix, stage='source'):
        if self.model_name in ['v_GRU', 'r_GRU', 'v_STGCN']:
            return self.loss_criterion(out, y)
        return self.calculate_loss(out, y, meta_graph, matrix, stage, graph_loss=False)

    def _forward_with_optional_ib(self, data_i, A_gnd, stage='train', u_tau=None):
        if self.ib_enabled:
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

    def checkpoint_state(self, epoch, optimizer=None, config=None, best_metric=None):
        return {
            "epoch": epoch,
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "config": config,
            "best_metric": best_metric,
            "use_pattern_ib": self.use_pattern_ib,
            "use_meta_ib": self.use_meta_ib
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
        
        # taskwise loss, precision
        task_losses = []
        for i in range(self.task_num):
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
                loss = self.combine_losses(pred_loss, ib_losses["pattern_ib_loss"], meta_ib_loss)
                grad = torch.autograd.grad(loss, self.model.parameters(), allow_unused=True, retain_graph=self.ib_enabled)
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
                loss_q = self.combine_losses(pred_loss_q, ib_losses_q["pattern_ib_loss"], meta_ib_loss)
                if self.ib_enabled:
                    total_pred_loss.append(pred_loss_q.detach().item())
                    total_pattern_ib_loss.append(ib_losses_q["pattern_ib_loss"].detach().item())
                    total_meta_ib_loss.append(meta_ib_loss.detach().item())
                    total_total_loss.append(loss_q.detach().item())
                
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

            if self.ib_enabled:
                u_tau, meta_ib_loss = self._encode_task_ib(data_i, deterministic=False)
                out,y,Ax,ib_losses = self.model(data_i, A_gnd, u_tau=u_tau, return_ib_loss=True)
                pred_loss = loss_fn(out, y)
                loss = self.combine_losses(pred_loss, ib_losses["pattern_ib_loss"], meta_ib_loss)
                total_pred_loss.append(pred_loss.detach().item())
                total_pattern_ib_loss.append(ib_losses["pattern_ib_loss"].detach().item())
                total_meta_ib_loss.append(meta_ib_loss.detach().item())
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
        return total_mse,total_rmse, total_mae, total_mape, total_loss

    def test_batch(self,start,end,source_dataset,stage = "test"):
        total_loss = []
        total_mae = []
        total_mse = []
        total_rmse = []
        total_mape = []
        
        with torch.no_grad():
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
                if self.ib_enabled:
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


    def finetuning(self, finetune_dataset, test_dataset, target_epochs):
        """
        finetunning stage in MAML
        """
        if target_epochs <= 0:
            print("[INFO] Skip finetune phase because target_epochs <= 0")
            return
        curr_time = datetime.datetime.now().strftime(
    "%Y%m%d-%H%M%S")
        if self.ib_enabled:
            model_path = Path(self.PatchFSL_cfg.get('checkpoint_dir', self.PatchFSL_cfg.get('ib_save_dir', './save/ib_runs')))
            checkpoint_prefix = self.PatchFSL_cfg.get('checkpoint_prefix', 'tpb_ib')
            target_city = self.PatchFSL_cfg.get('test_dataset', 'target')
            save_every = max(1, int(self.PatchFSL_cfg.get('save_every', 1)))
            save_best_only = bool(self.PatchFSL_cfg.get('save_best_only', False))
            overwrite_checkpoint = bool(self.PatchFSL_cfg.get('overwrite_checkpoint', False))
            enable_checkpoint = bool(self.PatchFSL_cfg.get('enable_checkpoint', True))
        else:
            model_path = Path('./save/meta_model/{}/{}/'.format(self.PatchFSL_cfg['data_list'],curr_time))
        
        if(not os.path.exists(model_path)):
            os.makedirs(model_path)
        print("Finetuned model saved in {}".format(model_path))
        optimizer = optim.Adam(self.model.parameters(), lr=self.meta_lr, weight_decay=1e-2)

        best_loss = 9999999999999.0
        best_model = None
        print("[INFO] Enter finetune phase")

        for i in range(target_epochs):
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
            
            mae_loss = np.mean(total_mae)
            if(mae_loss < best_loss):
                best_model = copy.deepcopy(self.model)
                best_loss = mae_loss
                if self.ib_enabled and enable_checkpoint:
                    save_checkpoint(
                        model_path / '{}_{}_best.pt'.format(checkpoint_prefix, target_city),
                        self.checkpoint_state(i, optimizer, self.PatchFSL_cfg.get('config', None), best_loss),
                        overwrite=overwrite_checkpoint
                    )
                else:
                    torch.save(self.model.state_dict(), model_path / 'finetuned_bestmodel.pt')
                print('Best model. Saved.')
            if self.ib_enabled and enable_checkpoint and (not save_best_only) and ((i + 1) % save_every == 0):
                state = self.checkpoint_state(i, optimizer, self.PatchFSL_cfg.get('config', None), best_loss)
                save_checkpoint(model_path / '{}_{}_last.pt'.format(checkpoint_prefix, target_city), state, overwrite=overwrite_checkpoint)
                save_checkpoint(model_path / '{}_{}_epoch{}.pt'.format(checkpoint_prefix, target_city, i), state, overwrite=overwrite_checkpoint)
            print('this epoch costs {:.5}s'.format(time.time()-time_1))

        print("[INFO] Enter test phase")
        length = test_dataset.__len__()
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

        
