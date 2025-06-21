import math
import numbers
import os
import sys
from collections import OrderedDict
from datetime import date

import h5py
import numpy as np
import torch
import random


from scipy.io import loadmat
from scipy.sparse import csr_matrix, diags
from sklearn.model_selection import KFold
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR, _LRScheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from typing import Callable, Dict
from torch.utils.tensorboard import SummaryWriter
try:
    from libs.utils import *
except:
    from utils import *

current_path = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.dirname(os.path.dirname(current_path))
MODEL_PATH = default(os.environ.get('MODEL_PATH'),
                     os.path.join(SRC_ROOT, 'models'))
DATA_PATH = default(os.environ.get('DATA_PATH'),
                    os.path.join(SRC_ROOT, 'data'))
FIG_PATH = default(os.environ.get('FIG_PATH'),
                   os.path.join(os.path.dirname(SRC_ROOT), 'figures'))
EPOCH_SCHEDULERS = ['ReduceLROnPlateau', 'StepLR', 'MultiplicativeLR',
                    'MultiStepLR', 'ExponentialLR', 'LambdaLR']
PI = math.pi
SEED = default(os.environ.get('SEED'), 1127802)
PHI_NAME = "phi_bm10bp1"
COEFF_NAME = "I_true_num4"

def pooling_2d(mat, kernel_size: tuple = (2, 2), method='mean', padding=False):
    '''Non-overlapping pooling on 2D data (or 2D data stacked as 3D array).

    mat: ndarray, input array to pool. (m, n) or (bsz, m, n)
    kernel_size: tuple of 2, kernel size in (ky, kx).
    method: str, 'max for max-pooling, 
                   'mean' for mean-pooling.
    pad: bool, pad <mat> or not. If no pad, output has size
           n//f, n being <mat> size, f being kernel size.
           if pad, output has size ceil(n/f), padding is nan
           so when computing mean the 0 is counted

    Return <result>: pooled matrix.

    Modified from https://stackoverflow.com/a/49317610/622119
    to handle the case of batch edge matrices
    CC BY-SA 3.0
    '''

    m, n = mat.shape[-2:]
    ky, kx = kernel_size

    def _ceil(x, y): return int(np.ceil(x/float(y)))

    if padding:
        ny = _ceil(m, ky)
        nx = _ceil(n, kx)
        size = mat.shape[:-2] + (ny*ky, nx*kx)
        sy = (ny*ky - m)//2
        sx = (nx*kx - n)//2
        _sy = ny*ky - m - sy
        _sx = nx*kx - n - sx

        mat_pad = np.full(size, np.nan)
        mat_pad[..., sy:-_sy, sx:-_sx] = mat
    else:
        ny = m//ky
        nx = n//kx
        mat_pad = mat[..., :ny*ky, :nx*kx]

    new_shape = mat.shape[:-2] + (ny, ky, nx, kx)

    if method == 'max':
        result = np.nanmax(mat_pad.reshape(new_shape), axis=(-3, -1))
    elif method == 'mean':
        result = np.nanmean(mat_pad.reshape(new_shape), axis=(-3, -1))
    else:
        raise NotImplementedError("pooling method not implemented.")

    return result


def train_batch_eit(model, loss_func, data, optimizer, lr_scheduler, device, grad_clip=0.99):
    optimizer.zero_grad()
    x, gradx = data["phi"].to(device), data["gradphi"].to(device)
    grid_c, grid = data['grid_c'].to(device), data['grid'].to(device)
    targets = data["targets"].to(device)

    # pos is for attention, grid is the finest grid
    out_ = model(x, gradx, pos=grid_c, grid=grid)
    out = out_['preds']

    # out is (b, n, n, 1)
    loss = loss_func(out, targets)

    loss.backward()
    optimizer.step()
    if lr_scheduler:
        lr_scheduler.step()

    return loss.item()

# def train_batch_ims(model, loss_func, data, optimizer, optimizer_, lr_scheduler, lr_scheduler_, device, grad_clip=0.99):
def train_batch_ims(model, loss_func, data, optimizer, lr_scheduler, device, grad_clip=0.99):

    optimizer.zero_grad()
    u_s_real = data["u_s_real"].to(device)
    u_s_imag = data["u_s_imag"].to(device)
    u_s = torch.stack([u_s_real, u_s_imag], dim=0)
    targets = data["targets"].to(device)
    
    out_ = model(u_s)
    out = out_['preds']
    
    loss_1 = loss_func(out, targets)

    loss_1.backward()
    optimizer.step()

    if lr_scheduler:
        lr_scheduler.step()


    for param in model.parameters():
        param.requires_grad = False

    # model[0].s[0].requires_grad = False
    # model[0].s[1].requires_grad = False
    # model[0].s[2].requires_grad = False

    for param in model[2].out.parameters():
        param.requires_grad = True

    optimizer.zero_grad()
    u_s_real = data["u_s_real"].to(device)
    u_s_imag = data["u_s_imag"].to(device)
    u_s = torch.stack([u_s_real, u_s_imag], dim=0)
    targets = data["targets"].to(device)
    
    out_ = model(u_s)
    out = out_['preds']
    
    loss_2 = loss_func(out, targets)

    loss_2.backward()
    optimizer.step()
    if lr_scheduler:
        lr_scheduler.step()

    for param in model.parameters():
        param.requires_grad = True
    
    if hasattr(model[0], 's') and isinstance(model[0].s, torch.nn.Parameter):
        model[0].s.requires_grad = False  # 设置model[0].s不需要计算梯度


    return loss_2.item()
    
def train_batch_ims_LBFGS(model, loss_func, data, optimizer, lr_scheduler, device, grad_clip=0.99):
    def closure():
        optimizer.zero_grad()
        
        u_s_real = data["u_s_real"].to(device)
        u_s_imag = data["u_s_imag"].to(device)
        u_s = torch.stack([u_s_real, u_s_imag], dim=0)
        targets = data["targets"].to(device)
        
        out_ = model(u_s)
        out = out_['preds']
        loss = loss_func(out, targets)
        loss.backward()
        return loss
    
    optimizer.step(closure)
    # for name, param in model[2].out.named_parameters():
    #     print(f"Parameter name: {name}")
    #     print(f"Parameter value:\n{param.data.cpu().numpy()}\n")
    
    # 返回损失值
    return closure().item()


# 进行一次优化步骤


# def validate_epoch_eit(model, metric_funcs, valid_loader, device):
#     model.eval()

#     if isinstance(metric_funcs, Callable):
#         metric_val = []
#         for _, data in enumerate(valid_loader):
#             with torch.no_grad():
#                 x, g = data["phi"].to(device), data["gradphi"].to(device)
#                 grid_c, grid = data['grid_c'].to(device), data['grid'].to(device)
#                 targets = data["targets"].to(device)
#                 out_ = model(x, g, pos=grid_c, grid=grid)
#                 preds = out_['preds']
#                 targets = data["targets"].to(device)
#                 weights = torch.ones_like(targets)
#                 metric = metric_funcs(preds, targets, weights=weights)

#                 metric = metric.cpu().numpy()
#                 metric_val.append(metric)
#         return dict(metric=np.mean(metric_val, axis=0))

#     elif isinstance(metric_funcs, Dict):
#         metric_names = metric_funcs.keys()
#         metric_vals = {m: [] for m in metric_names}
        
#         for _, data in enumerate(valid_loader):
#             with torch.no_grad():
#                 x, g = data["phi"].to(device), data["gradphi"].to(device)
#                 grid_c, grid = data['grid_c'].to(device), data['grid'].to(device)
#                 targets = data["targets"].to(device)
#                 out_ = model(x, g, pos=grid_c, grid=grid)
#                 preds = out_['preds']
#                 targets = data["targets"].to(device)
#                 weights = torch.ones_like(targets)

#                 for m in metric_names:
#                     metric_f = metric_funcs[m]
#                     metric = metric_f(preds, targets, weights=weights)
#                     metric = metric.cpu().numpy()
#                     metric_vals[m].append(metric)

#         for m in metric_names:
#             metric_vals[m] = np.mean(metric_vals[m], axis=0)

#         return metric_vals

def validate_epoch_ims_flag(model, metric_funcs, valid_loader, flag, device):
    model.eval()

    if isinstance(metric_funcs, Callable):
        metric_val = []
        for _, data in enumerate(valid_loader):
            with torch.no_grad():
                x_real= data["u_s_real"].to(device)
                x_imag= data["u_s_imag"].to(device)
                x = torch.stack([x_real, x_imag], dim=0)
                targets = data["targets"].to(device)
                out_ = model(x)
                preds = out_['preds']
                targets = data["targets"].to(device)
                weights = torch.ones_like(targets)
                metric = metric_funcs(preds, targets, weights=weights)

                metric = metric.cpu().numpy()
                metric_val.append(metric)
        return dict(metric=np.mean(metric_val, axis=0))

    elif isinstance(metric_funcs, Dict):
        metric_names = metric_funcs.keys()
        metric_vals = {m: [] for m in metric_names}
        for i, data in enumerate(valid_loader):
            with torch.no_grad():
                x_real = data["u_s_real"].to(device)
                x_imag = data["u_s_imag"].to(device)
                x = torch.stack([x_real, x_imag], dim=0)
                out_ = model(x)
                preds = out_['preds']
                targets = data["targets"].to(device)
            for j in range(100):
                flag_ = flag[j+i*100]
                with torch.no_grad():
                    pred = preds[j, ...]
                    target = targets[j, ...]
                    flag_pred = []
                    flag_target = []
                    for row, col in flag_:
                        pred_ = pred[row, col]
                        target_ = target[row, col]
                        flag_pred.append(pred_)
                        flag_target.append(target_)
                    flag_pred = torch.tensor(flag_pred).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                    flag_target = torch.tensor(flag_target).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                    weights = torch.ones_like(flag_target)
                
                    for m in metric_names:
                        metric_f = metric_funcs[m]
                        metric = metric_f(flag_pred, flag_target, weights=weights)
                        metric = metric.cpu().numpy()
                        metric_vals[m].append(metric)
        
        for m in metric_names:
            print(np.argmin(metric_vals[m]))
            metric_vals[m] = np.mean(metric_vals[m], axis=0)

        return metric_vals

def validate_epoch_ims(model, metric_funcs, valid_loader, device):
    model.eval()

    if isinstance(metric_funcs, Callable):
        metric_val = []
        for _, data in enumerate(valid_loader):
            with torch.no_grad():
                x_real= data["u_s_real"].to(device)
                x_imag= data["u_s_imag"].to(device)
                x = torch.stack([x_real, x_imag], dim=0)
                targets = data["targets"].to(device)
                out_ = model(x)
                preds = out_['preds']
                targets = data["targets"].to(device)
                weights = torch.ones_like(targets)
                metric = metric_funcs(preds, targets, weights=weights)

                metric = metric.cpu().numpy()
                metric_val.append(metric)
        return dict(metric=np.mean(metric_val, axis=0))

    elif isinstance(metric_funcs, Dict):
        metric_names = metric_funcs.keys()
        metric_vals = {m: [] for m in metric_names}
        for _, data in enumerate(valid_loader):
            with torch.no_grad():
                x_real = data["u_s_real"].to(device)
                x_imag = data["u_s_imag"].to(device)
                x = torch.stack([x_real, x_imag], dim=0)
                targets = data["targets"].to(device)
                out_ = model(x)
                preds = out_['preds']
                targets = data["targets"].to(device)
                weights = torch.ones_like(targets)
            
                for m in metric_names:
                    metric_f = metric_funcs[m]
                    metric = metric_f(preds, targets, weights=weights)
                    metric = metric.cpu().numpy()
                    metric_vals[m].append(metric)
        
        for m in metric_names:
            metric_vals[m] = np.mean(metric_vals[m], axis=0)

        return metric_vals

def run_train(model, loss_func, metric_func,
              train_loader, valid_loader,
            #   optimizer, optimizer_, lr_scheduler, lr_scheduler_,
              optimizer, lr_scheduler,
              train_batch=train_batch_ims,
              validate_epoch=validate_epoch_ims,
              epochs=10,
              visualization_path_training='./',
              visualization_path_validation='./',
              visualization_path_parameter='./',
              device="cuda",
              mode='min',
              tqdm_mode='batch',
              patience=10,
              grad_clip=0.999,
              start_epoch: int = 0,
              model_save_path=MODEL_PATH,
              save_mode='state_dict',
              model_name='model.pt',
              result_name='result.pkl'):
    loss_train = []
    loss_val = []
    loss_epoch = []
    lr_history = []
    # parameter_writers = {}
    it = 0

    if patience is None or patience == 0:
        patience = epochs
    start_epoch = start_epoch
    end_epoch = start_epoch + epochs
    best_val_metric = -np.inf if mode == 'max' else np.inf
    best_val_epoch = None
    save_mode = 'state_dict' if save_mode is None else save_mode
    stop_counter = 0
    is_epoch_scheduler = any(s in str(lr_scheduler.__class__)
                             for s in EPOCH_SCHEDULERS)
    tqdm_epoch = False if tqdm_mode == 'batch' else True
    with tqdm(total=end_epoch-start_epoch, disable=not tqdm_epoch) as pbar_ep:
        training_writer = SummaryWriter(log_dir=visualization_path_training)
        validation_writer = SummaryWriter(log_dir=visualization_path_validation)
        parameters_writer = SummaryWriter(log_dir=visualization_path_parameter)
        global_step = 0
        epoch_step = 0
        for epoch in range(start_epoch, end_epoch):
            # if epoch >= 0 and epoch < 10:
            # if epoch > 0:
            # with torch.no_grad():
                # lower_bound = max(-model[0].s, -0.999999)  # 确保s_value + a >= 0
                # upper_bound = min(1 - model[0].s, 0.999999)  # 确保s_value + a <= 1
                # a = random.uniform(lower_bound, upper_bound)
                # new_s = model[0].s + a
                # model[0].s.data = new_s
                    

                # lower_bounds = [max(-model[0].s[0], -0.999999),
                #                 max(-model[0].s[1], -0.999999),
                #                 max(-model[0].s[2], -0.999999)]
                # max_lower_bound = max(lower_bounds)
                # upper_bounds = [min(1 - model[0].s[0], 0.999999),
                #                 min(1 - model[0].s[1], 0.999999),
                #                 min(1 - model[0].s[2], 0.999999)]
                # min_upper_bound = min(upper_bounds)
                # a = random.uniform(max_lower_bound, min_upper_bound)
                
                # new_s_0 = model[0].s[0] + a 
                # new_s_1 = model[0].s[1] + a
                # new_s_2 = model[0].s[2] + a
                # model[0].s[0].data = new_s_0
                # model[0].s[1].data = new_s_1
                # model[0].s[2].data = new_s_2
        
            # if epoch == 10:
            #     model[0].s[0].data = torch.tensor(1.0, device=device)
            #     model[0].s[1].data = torch.tensor(1.0, device=device)
            #     model[0].s[2].data = torch.tensor(1.0, device=device)

            model.train()
            with tqdm(total=len(train_loader), disable=tqdm_epoch) as pbar_batch:
                for batch in train_loader:
                    if is_epoch_scheduler:
                        loss = train_batch(model, loss_func,
                                           batch, optimizer,
                                           None, device, grad_clip=grad_clip)
                    else:
                        loss = train_batch(model, loss_func,
                                           batch, 
                                        #    optimizer, optimizer_, lr_scheduler, lr_scheduler_, 
                                           optimizer, lr_scheduler, 
                                           device, grad_clip=grad_clip)

                    # training_writer.add_histogram(name, param, global_step=global_step)
                    # training_writer.add_histogram('s', param, global_step=global_step)
                    
                    for name, param in model.named_parameters():
                        scalar_dict = {}
                        if '0.s' in name:
                            scalar_dict[name] = param
                            parameters_writer.add_scalars('s', scalar_dict, global_step=global_step)
                    loss = np.array(loss)
                    training_writer.add_scalar('train', loss, global_step=global_step)
                    global_step +=1

                    loss_epoch.append(loss)
                    it += 1
                    lr = optimizer.param_groups[0]['lr']
                    lr_history.append(lr)
                    desc = f"epoch: [{epoch+1}/{end_epoch}]"
                    if loss.ndim == 0:  # 1 target loss
                        _loss_mean = np.mean(loss_epoch)
                        desc += f" loss: {_loss_mean:.3e}"
                    else:
                        _loss_mean = np.mean(loss_epoch, axis=0)
                        for j in range(len(_loss_mean)):
                            if _loss_mean[j] > 0:
                                desc += f" | loss {j}: {_loss_mean[j]:.3e}"
                    desc += f" | current lr: {lr:.3e}"
                    pbar_batch.set_description(desc)
                    pbar_batch.update()

            loss_train.append(_loss_mean)
            # loss_train.append(loss_epoch)
            loss_epoch = []

            val_result = validate_epoch(
                model, metric_func, valid_loader, device)
            validation_writer.add_scalar('validate', val_result["metric"], global_step=epoch_step)
            epoch_step +=1
            loss_val.append(val_result["metric"])
            val_metric = val_result["metric"].sum()

            if mode == 'max':
                saving_criterion = (val_metric > best_val_metric)
            elif mode == 'min':
                saving_criterion = (val_metric < best_val_metric)

            if saving_criterion:
                best_val_epoch = epoch
                best_val_metric = val_metric
                stop_counter = 0
                if save_mode == 'state_dict':
                    torch.save(model.state_dict(), os.path.join(
                        model_save_path, model_name))
                else:
                    torch.save(model, os.path.join(
                        model_save_path, model_name))
                best_model_state_dict = {
                    k: v.to('cpu') for k, v in model.state_dict().items()}
                best_model_state_dict = OrderedDict(best_model_state_dict)

            else:
                stop_counter += 1

            if lr_scheduler and is_epoch_scheduler:
                if 'ReduceLROnPlateau' in str(lr_scheduler.__class__):
                    lr_scheduler.step(val_metric)
                else:
                    lr_scheduler.step()

            if stop_counter > patience:
                print(f"Early stop at epoch {epoch}")
                break
            if val_result["metric"].ndim == 0:
                desc = color(
                    f"| val metric: {val_metric:.3e} ", color=Colors.blue)
            else:
                desc = color(f"| ", color=Colors.blue)
                for i, _m in enumerate(val_result["metric"]):
                    desc += color(f"val metric {i+1}: {_m:.3e} ",
                                  color=Colors.blue)

            desc += color(
                f"| best val: {best_val_metric:.3e} at epoch {best_val_epoch+1}", color=Colors.yellow)
            desc += color(f" | early stop: {stop_counter} ", color=Colors.red)
            desc += color(f" | current lr: {lr:.3e}", color=Colors.magenta)
            if not tqdm_epoch:
                tqdm.write("\n"+desc+"\n")
            else:
                desc_ep = color("", color=Colors.green)
                if _loss_mean.ndim == 0:  # 1 target loss
                    desc_ep += color(f"| loss: {_loss_mean:.3e} ",
                                     color=Colors.green)
                else:
                    for j in range(len(_loss_mean)):
                        if _loss_mean[j] > 0:
                            desc_ep += color(
                                f"| loss {j}: {_loss_mean[j]:.3e} ", color=Colors.green)
                desc_ep += desc
                pbar_ep.set_description(desc_ep)
                pbar_ep.update()
            # if epoch >= 0 and epoch < 10:
            # if epoch > 0:
            # with torch.no_grad():
                # new_s_ = model[0].s - a
                # model[0].s.data = new_s_

                # new_s_0_ = model[0].s[0] - a
                # new_s_1_ = model[0].s[1] - a
                # new_s_2_ = model[0].s[2] - a
                # model[0].s[0].data = new_s_0_
                # model[0].s[1].data = new_s_1_
                # model[0].s[2].data = new_s_2_

                # model[0].s =nn.Parameter(new_s_, requires_grad=True)

            result = dict(
                best_val_epoch=best_val_epoch,
                best_val_metric=best_val_metric,
                loss_train=np.asarray(loss_train),
                loss_val=np.asarray(loss_val),
                lr_history=np.asarray(lr_history),
                # best_model=best_model_state_dict,
                optimizer_state=optimizer.state_dict()
            )
            save_pickle(result, os.path.join(model_save_path, result_name))
    return result

class UnitGaussianNormalizer:
    def __init__(self, eps=1e-5):
        super().__init__()
        '''
        modified from utils3.py in 
        https://github.com/zongyi-li/fourier_neural_operator
        Changes:
            - .to() has a return to polymorph the torch behavior
            - naming convention changed to sklearn scalers 
        '''
        self.eps = eps

    def fit_transform(self, x):
        self.mean = x.mean(0)
        self.std = x.std(0)
        return (x - self.mean) / (self.std + self.eps)

    def transform(self, x):
        return (x - self.mean) / (self.std + self.eps)

    def inverse_transform(self, x):
        return (x * (self.std + self.eps)) + self.mean

    def to(self, device):
        if torch.is_tensor(self.mean):
            self.mean = self.mean.float().to(device)
            self.std = self.std.float().to(device)
        else:
            self.mean = torch.from_numpy(self.mean).float().to(device)
            self.std = torch.from_numpy(self.std).float().to(device)
        return self

    def cuda(self, device=None):
        assert torch.is_tensor(self.mean)
        self.mean = self.mean.float().cuda(device)
        self.std = self.std.float().cuda(device)
        return self

    def cpu(self):
        assert torch.is_tensor(self.mean)
        self.mean = self.mean.float().cpu()
        self.std = self.std.float().cpu()
        return self


class EITDataset(Dataset):
    def __init__(self,
                 data_path=DATA_PATH,
                 file_type='h5',
                 phi_name=PHI_NAME,
                 coeff_name=COEFF_NAME,
                 part_idx: list = [4],
                 index_channel = None,
                 shuffle_channel=False,
                 normalizer_x=None,
                 normalization=False,
                 subsample: int = 1,
                 subsample_attn: int = 4,
                 subsample_method='nearest',
                 channel=5,
                 train_data=True,
                 train_len=0.8,
                 valid_len=0.2,
                 debug=False,
                 online_grad=True,
                 return_grad=True,
                 return_boundary=True,
                 random_state=1127802):
        '''
        EIT data from Fan-Bohorquez-Ying 2019, Guo-Jiang 2020
        '''
        assert max(part_idx) <= 6
        self.data_path = data_path
        self.file_type = file_type
        self.phi_name = phi_name
        self.coeff_name = coeff_name
        self.part_idx = part_idx
        self.parts = ['part'+str(i) for i in part_idx]
        self.shuffle_channel = shuffle_channel 
        self.index_channel = index_channel
        self.n_grid_fine = 64  # finest resolution along x-, y- dim
        self.subsample_attn = subsample_attn  # subsampling for attn
        self.subsample = subsample  # subsampling for input and output
        self.subsample_method = subsample_method  # 'interp' or 'nearest'
        self.channel = channel
        self.n_grid = int(((self.n_grid_fine - 1)/self.subsample) + 1)
        self.n_grid_coarse = int(
            ((self.n_grid_fine - 1)/self.subsample_attn) + 1)
        self.h = 1/self.n_grid_fine # grad 1st then subsample
        self.train_data = train_data
        self.train_len = train_len
        self.valid_len = valid_len
        self.normalization = normalization
        self.normalizer_x = normalizer_x  
        self.random_state = random_state
        self.return_grad = return_grad
        self.online_grad = online_grad
        self.return_boundary = return_boundary
        self.eps = 1e-8
        self.debug = debug
        if self.data_path is not None:
            self._initialize()

    def __len__(self):
        return self.n_samples

    def _initialize(self):
        get_seed(self.random_state, printout=False)
        with timer(f"Loading parts {'+'.join([str(i) for i in self.part_idx]) }"):
            self._get_files()
            phi, a = self.load_example()  # (N, n, n) and  (N, C, n, n)
            gc.collect()

        self.n_samples = len(a)

        self.phi, self.gradphi, self.targets = self.preprocess(phi, a)

        self.grid_c = self.get_grid(self.n_grid_coarse)
        self.grid = self.get_grid(self.n_grid_fine,
                                  subsample=self.subsample,
                                  return_boundary=self.return_boundary)

        if self.train_data and self.normalization:
            self.normalizer_x = UnitGaussianNormalizer()
            self.normalizer_y = UnitGaussianNormalizer()
            self.phi = self.normalizer_x.fit_transform(self.phi)

            if self.return_boundary:
                _ = self.normalizer_y.fit_transform(x=self.targets)
            else:
                _ = self.normalizer_y.fit_transform(
                    x=self.targets[:, 1:-1, 1:-1, :])
        elif self.normalization:
            self.phi = self.normalizer_x.transform(self.phi)

    def _get_files(self):
        data_files = find_files(self.phi_name, path=self.data_path)
        target_files = find_files(self.coeff_name, path=self.data_path)
        t = '.'+self.file_type
        data_file = []
        target_file = []
        for p in self.parts:
            data_file += [f for f in data_files if p in f and t in f]
            target_file += [f for f in target_files if p in f and t in f]
        self.data_file = sorted(data_file)
        self.target_file = sorted(target_file)

    def load_example(self):
        '''load example in mat files'''
        data_file, target_file = self.data_file, self.target_file

        a = []
        for f in target_file:
            if self.file_type == 'mat':
                a_ = loadmat(f)
                n = self.n_grid_fine
                assert a_["I_true"].shape[-1] == n**2
                a_ = a_["I_true"].reshape(-1, n, n)

            elif self.file_type == 'h5':
                a_ = h5py.File(f, mode="r")
                a_ = a_.get('I_true')
                # a.append(a_[()])
            a.append(a_[()].transpose(0, 2, 1))

        a = np.concatenate(a, axis=0)

        if self.debug:
            data_len = int(0.1*len(a))
            a = a[:data_len]
        else:
            data_len = self.get_data_len(len(a))
            if self.train_data:
                a = a[:data_len]
            else:
                a = a[-data_len:]

        u = []
        for _, d in enumerate(data_file):
            u_ = h5py.File(d, mode="r")
            key = list(u_.keys())[0]
            u_ = u_.get(key)
            u.append(u_[()])
        
        if self.shuffle_channel and self.index_channel is None:
            self.index_channel = np.random.randint(u[0].shape[1], size=self.channel)
        elif not self.shuffle_channel:
            self.index_channel = np.arange(self.channel)
        u = np.concatenate([x[:, self.index_channel, ...] for x in u], axis=0)

        if self.train_data:
            u = u[:data_len]
        else:
            u = u[-data_len:]

        return u, a

    def get_data_len(self, len_data):
        if self.train_data:
            if self.train_len <= 1:
                train_len = int(self.train_len*len_data)
            elif 1 < self.train_len < len_data:
                train_len = self.train_len
            else:
                train_len = int(0.8*len_data)
            return train_len
        else:
            if self.valid_len <= 1:
                valid_len = int(self.valid_len*len_data)
            elif 1 < self.valid_len < len_data:
                valid_len = self.valid_len
            else:
                valid_len = int(0.2*len_data)
            return valid_len

    def preprocess(self, u, a):
        # input is (N, C, 201, 201)
        bsz = a.shape[0]
        n_grid_fine = self.n_grid_fine
        s = self.subsample
        h = self.h
        n = self.n_grid

        if s > 1 and self.subsample_method == 'nearest':
            a = a[:, ::s, ::s]
        elif s > 1 and self.subsample_method in ['interp', 'linear', 'average']:
            a = pooling_2d(a,
                           kernel_size=(s, s),
                           padding=True)
        a = a.reshape(bsz, n, n, 1)

        if self.return_grad and not self.online_grad:
            gradu = self.get_grad(u, h)  # u is (N, C, n, n)
            gradu = gradu[..., ::s, ::s]
            gradu = gradu.transpose((0, 2, 3, 1)) # (N, n, n, C)
        else:
            gradu = np.zeros((bsz, ))  # placeholder

        u = u[..., ::s, ::s]
        u = u.transpose((0, 2, 3, 1)) # (N, n, n, C)
        return u, gradu, a

    @staticmethod
    def get_grad(f, h):
        '''
        h: mesh size
        n: grid size
        separate input for online grad generating
        input f: (N, C, n, n)
        '''
        bsz, N_C = f.shape[:2]
        # n = int(((n - 1)/s) + 1)
        
        fx, fy = [], []
        for i in range(N_C):
            '''smaller mem footprint'''
            _fx, _fy = EITDataset.central_diff(f[:, i], h)
            fx.append(_fx)
            fy.append(_fy)
        gradf = np.stack(fx+fy, axis=1)  # (N, 2*C, n, n)
        return gradf

    @staticmethod
    def central_diff(f, h, mode='constant', padding=True):
        """
        mode: see
        https://numpy.org/doc/stable/reference/generated/numpy.pad.html
        """
        # x: (batch, n, n)
        # b = x.shape[0]
        if padding:
            f = np.pad(f, ((0, 0), (1, 1), (1, 1)),
                       mode=mode, constant_values=0)
        d, s = 2, 1  # dilation and stride
        grad_x = (f[:, d:, s:-s] - f[:, :-d, s:-s]) / \
            d  # (N, S_x, S_y)
        grad_y = (f[:, s:-s, d:] - f[:, s:-s, :-d]) / \
            d  # (N, S_x, S_y)

        return grad_x/h, grad_y/h

    @staticmethod
    def get_grid(n_grid, subsample=1, return_boundary=True):
        x = np.linspace(0, 1, n_grid)
        y = np.linspace(0, 1, n_grid)
        x, y = np.meshgrid(x, y)
        s = subsample

        if return_boundary:
            x = x[::s, ::s]
            y = y[::s, ::s]
        else:
            x = x[::s, ::s][1:-1, 1:-1]
            y = y[::s, ::s][1:-1, 1:-1]
        grid = np.stack([x, y], axis=-1)
        # grid is DoF, excluding boundary (n, n, 2), or (n-2, n-2, 2)
        return grid

    def __getitem__(self, index):
        '''
        Outputs:
            - grid: x-, y- coords
            - grid_c: coarse grid
            - a: diffusion coeff
            - u: solution
            - gradu: gradient of solution
        '''
        pos_dim = 2
        # uniform grid for all samples (n_s*n_s, 2)
        grid_c = self.grid_c.reshape(-1, pos_dim)
        # uniform grid fine for all samples (n, n, 2)
        if self.subsample_attn is None:
            grid_c = torch.tensor([1.0])  # place holder
        else:
            grid_c = torch.from_numpy(grid_c)  # (n_s*n_s, 2)

        grid = torch.from_numpy(self.grid)  # fine grid (n, n, 2)

        phi_ = self.phi[index]
        phi = torch.from_numpy(phi_)
        targets = torch.from_numpy(self.targets[index])

        if self.return_grad and self.online_grad:
            phi_ = phi_[None, ...].transpose(0, 3, 1, 2)
            # phi_ (1, C, n, n)
            gradphi = self.get_grad(phi_, self.h)
            gradphi = gradphi.squeeze().transpose(1, 2, 0) #(n, n, 2*C)
            gradphi = torch.from_numpy(gradphi)
        elif self.return_grad:
            gradphi = torch.from_numpy(self.gradphi[index])
        else:
            gradphi = torch.tensor(float('nan'))

        return dict(phi=phi.float(),
                    gradphi=gradphi.float(),
                    grid_c=grid_c.float(),
                    grid=grid.float(),
                    targets=targets.float(),
                    )
#################################################################################################################

if __name__ == '__main__':
    dataset = EITDataset(part_idx=[1, 2],
                         file_type='h5',
                         subsample=1, 
                         channel=3,
                         return_grad=True,
                         online_grad=False,
                         train_data=True,
                         debug=False)
    loader = DataLoader(dataset,
                        batch_size=10,
                        shuffle=True,
                        drop_last=True,
                        pin_memory=True)
    sample = next(iter(loader))
    phi = sample['phi']
    gradphi = sample['gradphi']
    targets = sample['targets']
    print(phi.size(), gradphi.size(), targets.size())
