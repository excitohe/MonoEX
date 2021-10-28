from functools import partial

import torch.nn as nn
import torch.optim as optim

from .fastai_optim import OptimWrapper
from .fastai_sched import CosineWarmupLR, OneCycle


def get_model_params(model, cfg):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        key_lr = [cfg.SOLVER.BASE_LR]
        if "bias" in key:
            key_lr.append(cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR)
        params += [{"params": [value], "lr": max(key_lr)}]
    return params


def build_optimizer(model, cfg):
    if cfg.SOLVER.OPTIMIZER != "AdamOneCycle":
        model_params = get_model_params(model, cfg)

    if cfg.SOLVER.OPTIMIZER == "Adam":
        optimizer = optim.Adam(
            model_params, lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY, betas=(0.9, 0.99)
        )
    elif cfg.SOLVER.OPTIMIZER == "AdamW":
        optimizer = optim.AdamW(
            model_params, lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY, betas=(0.9, 0.99)
        )
    elif cfg.SOLVER.OPTIMIZER == 'SGD':
        optimizer = optim.SGD(
            model_params, lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY, momentum=cfg.SOLVER.MOMENTUM
        )
    elif cfg.SOLVER.OPTIMIZER == 'adam_onecycle':

        def children(m: nn.Module):
            return list(m.children())

        def num_children(m: nn.Module) -> int:
            return len(children(m))

        flatten_model = lambda m: sum(map(flatten_model, m.children()), []) if num_children(m) else [m]
        get_layer_groups = lambda m: [nn.Sequential(*flatten_model(m))]

        optimizer = OptimWrapper.create(
            partial(optim.Adam, betas=(0.9, 0.99)),
            cfg.SOLVER.BASE_LR,
            get_layer_groups(model),
            wd=cfg.SOLVER.WEIGHT_DECAY,
            true_wd=True,
            bn_wd=True
        )
    else:
        raise ValueError(f"Unsupport optimizer: {cfg.SOLVER.OPTIMIZER}")
    return optimizer


def build_scheduler(optimizer, cfg, last_epoch=-1):
    decay_steps = cfg.SOLVER.STEPS

    def lr_lambda(cur_epoch):
        cur_decay = 1
        for decay_step in decay_steps:
            if cur_epoch >= decay_step:
                cur_decay = cur_decay * cfg.SOLVER.LR_DECAY
        return max(cur_decay, cfg.SOLVER.LR_CLIP / cfg.SOLVER.BASE_LR)

    warmup_scheduler = None

    if cfg.SOLVER.SCHEDULER == "MultiStepLR":
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, cfg.SOLVER.STEPS, gamma=0.1, last_epoch=last_epoch)
    elif cfg.SOLVER.SCHEDULER == "OneCycleLR":
        scheduler = OneCycle(
            optimizer, cfg.SOLVER.MAX_ITERATION, cfg.SOLVER.BASE_LR, list(cfg.SOLVER.MOMS), cfg.SOLVER.DIV_FACTOR,
            cfg.SOLVER.PCT_START
        )
    elif cfg.SOLVER.SCHEDULER == "LambdaLR":
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)
        if cfg.SOLVER.LR_WARMUP:
            warmup_scheduler = CosineWarmupLR(
                optimizer, T_max=cfg.SOLVER.WARMUP_STEPS, eta_min=cfg.SOLVER.BASE_LR / cfg.SOLVER.DIV_FACTOR
            )
    else:
        raise ValueError(f"Unsupport scheduler: {cfg.SOLVER.SCHEDULER}")

    return scheduler, warmup_scheduler
