from contextlib import redirect_stdout
import os
from typing import Dict, Tuple, Union
from distutils.util import strtobool
from .constant import *
import torch
import torch.nn as nn
from torch.optim import optimizer, lr_scheduler, Adam, SGD
from adabelief_pytorch import AdaBelief


def init_weights(
    net: nn.Module,
    init_type: str = INIT_TYPE_NORMAL,
    gain: float = 0.02
) -> None:
    def init_func(m: nn.Module):
        if isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, gain)
            nn.init.constant_(m.bias.data, 0.0)
        elif hasattr(m, 'weight'):
            if init_type == INIT_TYPE_NORMAL:
                nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == INIT_TYPE_XAVIER:
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == INIT_TYPE_KAIMING:
                nn.init.kaiming_normal_(m.weight.data)
            elif init_type == INIT_TYPE_ORTHOGONAL:
                nn.init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(
                    f'Initialization method "{init_type}" is not implemented.')
        elif hasattr(m, 'bias'):
            nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


def get_optimizer(
    net: nn.Module,
    type: str,
    config: Dict[str, float]
) -> optimizer.Optimizer:
    """Optimizer Initialization

    Args:
        net (nn.Module): Target module.
        type (str): ['adam', 'sgd', 'adabelief']
        config (Dict[str, float]): {lr: FLOAT, adam:{beta1: FLOAT, beta2: FLOAT}, ...}

    Raises:
        KeyError: [description]
        NotImplementedError: [description]

    Returns:
        Optimizer: Optimizer
    """
    params: Dict[str, float] = config.get(type)
    if params is None:
        raise KeyError(f'Key "{type}" does not exist in "config".')
    for key in (config.keys() - {CONFIG_LR, type}):
        config.pop(key)

    if type == OPTIM_TYPE_ADAM:
        return Adam(net.parameters(), lr=config[CONFIG_LR], betas=(params[CONFIG_BETA1], params[CONFIG_BETA2]))
    elif type == OPTIM_TYPE_SGD:
        return SGD(net.parameters(), lr=config[CONFIG_LR], momentum=params[CONFIG_MOMENTUM], nesterov=True)
    elif type == OPTIM_TYPE_ADABELIEF:
        with redirect_stdout(open(os.devnull, 'w')):
            return AdaBelief(net.parameters(), lr=config[CONFIG_LR], betas=(params[CONFIG_BETA1], params[CONFIG_BETA2]), print_change_log=False)
    else:
        raise NotImplementedError(f'Optimizer "{type}" is not implemented.')


def get_scheduler(
    optimizer: optimizer.Optimizer,
    type: str,
    config: Dict[str, float],
    steps_per_epoch: int
) -> Union[lr_scheduler.StepLR, lr_scheduler.LambdaLR, lr_scheduler.ReduceLROnPlateau, lr_scheduler.CosineAnnealingLR, lr_scheduler.CyclicLR]:

    params: Dict[str, Union[int, float]] = config.get(type)
    if params is None:
        raise KeyError(f'Key "{type}" does not exist in "config".')
    for key in (config.keys() - {type}):
        config.pop(key)

    if type == LR_POLICY_LAMBDA:
        def lambda_rule(epoch):
            return 1.0 - max(0, epoch + params[CONFIG_EPOCH_COUNT] - params[CONFIG_NITER])
        return lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif type == LR_POLICY_STEP:
        return lr_scheduler.StepLR(optimizer, step_size=params[CONFIG_NITER_DECAY], gamma=params[CONFIG_GAMMA])
    elif type == LR_POLICY_PLATEAU:
        return lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=params[CONFIG_FACTOR], threshold=params[CONFIG_THRESHOLD], patience=params[CONFIG_PATIENCE]
        )
    elif type == LR_POLICY_COS:
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=params[CONFIG_NITER], eta_min=params[CONFIG_ETA_MIN])
    elif type == LR_POLICY_CLR:
        if isinstance(optimizer, SGD):
            cycle_momentum: bool = True
        else:
            cycle_momentum: bool = False
        return lr_scheduler.CyclicLR(
            optimizer, base_lr=params[CONFIG_BASE_LR], max_lr=params[CONFIG_MAX_LR], step_size_up=steps_per_epoch*2, cycle_momentum=cycle_momentum
        )
    else:
        raise NotImplementedError(f'Scheduler "{type}" is not implemented.')


def init_optimizer(
    net: nn.Module,
    optimizer_type: str,
    scheduler_type: str,
    config: Dict[str, float],
    steps_per_epoch: int
) -> Tuple[optimizer.Optimizer, Union[lr_scheduler.StepLR, lr_scheduler.LambdaLR, lr_scheduler.ReduceLROnPlateau, lr_scheduler.CosineAnnealingLR, lr_scheduler.CyclicLR]]:
    optm = get_optimizer(net=net, type=optimizer_type,
                         config=config[CONFIG_OPTIMIZER])
    sche = get_scheduler(optimizer=optm, type=scheduler_type,
                         config=config[CONFIG_SCHEDULER], steps_per_epoch=steps_per_epoch)
    return optm, sche


def update_lr(
    scheduler: Union[lr_scheduler.StepLR, lr_scheduler.LambdaLR, lr_scheduler.ReduceLROnPlateau, lr_scheduler.CosineAnnealingLR, lr_scheduler.CyclicLR],
    optimizer: optimizer.Optimizer,
    metric: torch.Tensor = None
) -> float:
    """Update learning rate

    Args:
        scheduler (Union[StepLR, LambdaLR, ReduceLROnPlateau, CosineAnnealingLR, CyclicLR]): Scheduler
        optimizer (Optimizer): Optimizer
        metric (torch.Tensor, optional): Metric for 'ReduceLROnPlateau'. Defaults to None.

    Returns:
        float: Learning Rate
    """
    if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
        scheduler.step(metric)
    else:
        scheduler.step()
    lr = optimizer.param_groups[0]['lr']
    return lr


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Batch quaternion multiplication

    Args:
        q1 (torch.Tensor): shape=(N, 4), [qx, qy, qz, qw]
        q2 (torch.Tensor): shape=(N, 4), [qx, qy, qz, qw]

    Returns:
        torch.Tensor: shape=(N, 4), [qx, qy, qz, qw]
    """

    q: torch.Tensor = torch.zeros(q1.shape[0], 4, device=q1.device)

    q[:, Q_IDX.w] = q2[:, Q_IDX.w] * q1[:, Q_IDX.w] - q2[:, Q_IDX.x] * q1[:,
                                                                          Q_IDX.x] - q2[:, Q_IDX.y] * q1[:, Q_IDX.y] - q2[:, Q_IDX.z] * q1[:, Q_IDX.z]
    q[:, Q_IDX.x] = q2[:, Q_IDX.w] * q1[:, Q_IDX.x] + q2[:, Q_IDX.x] * q1[:,
                                                                          Q_IDX.w] - q2[:, Q_IDX.y] * q1[:, Q_IDX.z] + q2[:, Q_IDX.z] * q1[:, Q_IDX.y]
    q[:, Q_IDX.y] = q2[:, Q_IDX.w] * q1[:, Q_IDX.y] + q2[:, Q_IDX.x] * q1[:,
                                                                          Q_IDX.z] + q2[:, Q_IDX.y] * q1[:, Q_IDX.w] - q2[:, Q_IDX.z] * q1[:, Q_IDX.x]
    q[:, Q_IDX.z] = q2[:, Q_IDX.w] * q1[:, Q_IDX.z] - q2[:, Q_IDX.x] * q1[:,
                                                                          Q_IDX.y] + q2[:, Q_IDX.y] * q1[:, Q_IDX.x] + q2[:, Q_IDX.z] * q1[:, Q_IDX.w]

    return q


def quat_inv(q: torch.Tensor) -> torch.Tensor:
    dst: torch.Tensor = q.clone()
    dst[:, :3] *= -1.0
    return dst


def str2bool(arg: str) -> bool:
    return bool(strtobool(arg))
