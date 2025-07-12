# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch
import time
import timm
from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
from loss import SetCriterion
import utils


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler=None,
                    max_norm: float = 0, args=None):
    model.train()
    criterion.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.7f}'))
    header = f'Epoch: [{epoch}]'

    print_freq = 1

    print("[DEBUG] Starting training loop...")

    for (samples, lidars), targets in metric_logger.log_every(data_loader, print_freq, header):

        print(f"[DEBUG] DataLoader length: {len(data_loader)}")

        samples = samples.to(device, non_blocking=True)
        lidars = lidars.to(device, non_blocking=True)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        print("⭐ Ready to forward model")

        outputs = model(samples, lidars)

        loss_dict = criterion(outputs, targets)

        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        if isinstance(loss_scaler, timm.utils.NativeScaler):
            loss_scaler(losses, optimizer, clip_grad=max_norm, parameters=model.parameters())
        else:
            losses.backward()
            if max_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        loss_value = losses.item()
        loss_dict_reduced = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()}

        lr = optimizer.param_groups[0]["lr"]
        print(
            # f"[Epoch {epoch}][Batch {data_iter_step}] "
            f"Loss: {loss_value:.4f}, "
            f"LR: {lr:.7f}, "
            f"Loss components: {loss_dict_reduced}"
        )

        print(f"[Epoch {epoch}] Batch Loss: {loss_value:.4f}, LR: {optimizer.param_groups[0]['lr']:.2e}")
        for k, v in loss_dict_reduced.items():
            print(f"  {k}: {v:.4f}")

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            if args.if_continue_inf:
                continue
            else:
                sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=float(f"{optimizer.param_groups[0]['lr']:.2e}"))

        # metric_logger.update(loss=loss_value, **loss_dict)
        # # metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        # metric_logger.update(lr=float(f"{optimizer.param_groups[0]['lr']:.2e}"))

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

# def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
#                     data_loader: Iterable, optimizer: torch.optim.Optimizer,
#                     device: torch.device, epoch: int, loss_scaler=None,
#                     max_norm: float = 0, args=None):
#     model.train()
#     criterion.train()
#
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.7f}'))
#     header = f'Epoch: [{epoch}]'
#
#     print_freq = 1
#
#     end = time.time()
#
#     for (samples, lidars), targets in metric_logger.log_every(data_loader, print_freq, header):
#         data_time = time.time() - end
#
#         samples = samples.to(device, non_blocking=True)
#         lidars = lidars.to(device, non_blocking=True)
#         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
#
#         outputs = model(samples, lidars)
#
#         if isinstance(outputs, (tuple, list)):
#             outputs = outputs[0]
#
#         loss_dict = criterion(outputs, targets)
#         weight_dict = criterion.weight_dict
#         losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
#
#         optimizer.zero_grad()
#
#         if isinstance(loss_scaler, timm.utils.NativeScaler):
#             loss_scaler(losses, optimizer, clip_grad=max_norm, parameters=model.parameters())
#         else:
#             losses.backward()
#             if max_norm is not None:
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
#             optimizer.step()
#
#         loss_value = losses.item()
#         loss_dict_reduced = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()}
#
#         print(f"[Epoch {epoch}] Batch Loss: {loss_value:.4f}, LR: {optimizer.param_groups[0]['lr']:.2e}")
#         for k, v in loss_dict_reduced.items():
#             print(f"  {k}: {v:.4f}")
#
#         if not math.isfinite(loss_value):
#             print(f"Loss is {loss_value}, stopping training")
#             if args.if_continue_inf:
#                 continue
#             else:
#                 sys.exit(1)
#
#         batch_time = time.time() - end
#         end = time.time()
#
#         metric_logger.update(loss=loss_value, **loss_dict_reduced)
#         metric_logger.update(lr=optimizer.param_groups[0]["lr"])
#         metric_logger.update(time=batch_time, data=data_time)
#
#     metric_logger.synchronize_between_processes()
#     print("Averaged stats:", metric_logger)
#     return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(model, criterion, data_loader, device):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    for (samples, lidars), targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device, non_blocking=True)
        lidars = lidars.to(device, non_blocking=True)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples, lidars)

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        metric_logger.update(loss=losses.item(), **loss_dict)

    metric_logger.synchronize_between_processes()
    print('Averaged evaluation stats:', metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
