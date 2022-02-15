from typing import Any, NamedTuple
import torch
from torch import Tensor
from enum import IntEnum


class PMOD_OUTPUT(NamedTuple):
    segmentation: Tensor
    depth: Tensor
    aux1: Tensor
    aux2: Tensor


class ADAPNET_DECODER_OUTPUT(NamedTuple):
    output: Tensor
    aux1: Tensor
    aux2: Tensor


class DISCRIMINATOR_OUTPUT(NamedTuple):
    macro: Tensor
    micro: Tensor


class LOSS_WEIGHT(NamedTuple):
    loss: torch.nn.Module
    weight: float

    def to(self, device: torch.device):
        self.loss.to(device)
        return self

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.loss(*args, **kwds) * self.weight


class Q_IDX(IntEnum):
    x: int = 0
    y: int = 1
    z: int = 2
    w: int = 3


class AXIS_IDX(IntEnum):
    x: int = 0
    y: int = 1
    z: int = 2


class METRIC_SUM_OUTPUT(NamedTuple):
    sum: Tensor
    count: Tensor


class METRIC_RESULT:
    def __init__(self, name: str = '', use_class: bool = False, use_thresholds: bool = False, all_tag: str = 'All') -> None:
        self.name: str = name
        self.use_class: bool = use_class
        self.use_thresholds: bool = use_thresholds
        self.all_tag: str = all_tag

        self.all_metric: Tensor = None
        self.class_metric: Tensor = None
        self.threshold_metric: Tensor = None

        self.counts_metric: Tensor = None
        self.sum_class_metric: Tensor = None
        self.sum_threshold_metric: Tensor = None

    def add(self, count_metric: Tensor = None, class_metric: Tensor = None, threshold_metric: Tensor = None):
        if isinstance(count_metric, Tensor):
            if self.counts_metric is None:
                self.counts_metric = torch.zeros_like(count_metric)
            self.counts_metric = self.counts_metric + count_metric

        if isinstance(class_metric, Tensor):
            tmp_class_metric = class_metric
        elif isinstance(self.class_metric, Tensor):
            tmp_class_metric = self.class_metric
        else:
            tmp_class_metric = None
        if tmp_class_metric is not None:
            if self.sum_class_metric is None:
                self.sum_class_metric = torch.zeros_like(tmp_class_metric)
            self.sum_class_metric = self.sum_class_metric + tmp_class_metric

        if isinstance(threshold_metric, Tensor):
            tmp_threshold_metric = threshold_metric
        elif isinstance(self.threshold_metric, Tensor):
            tmp_threshold_metric = self.threshold_metric
        else:
            tmp_threshold_metric = None
        if tmp_threshold_metric is not None:
            if self.sum_threshold_metric is None:
                self.sum_threshold_metric = torch.zeros_like(
                    tmp_threshold_metric)
            self.sum_threshold_metric = self.sum_threshold_metric + tmp_threshold_metric


class METRIC_BEST(NamedTuple):
    value: float
    path: str = None
    epoch: int = 0


def arg_hyphen(arg: str):
    return arg.replace('_', '-')


BASE_CHANNELS = 64
SKIP_CHANNELS = 24

# Train: Training
ARG_TAG = 'tag'
ARG_TRAIN_DL_CONFIG = 'train_dl_config'
ARG_VAL_DL_CONFIG = 'val_dl_config'
ARG_BLOCK_SIZE = 'block_size'
ARG_TRAIN_DATA = 'train_data'
ARG_VAL_DATA = 'val_data'
ARG_EPOCHS = 'epochs'
ARG_EPOCH_START_COUNT = 'epoch_start_count'
ARG_STEPS_PER_EPOCH = 'steps_per_epoch'
ARG_PROJECTED_POSITION_AUGMENTATION = 'projected_position_augmentation'
ARG_TR_ERROR_RANGE = 'tr_error_range'
ARG_ROT_ERROR_RANGE = 'rot_error_range'
ARG_AUTO_EVALUATION = 'auto_evaluation'
# Train: Network
ARG_BATCH_SIZE = 'batch_size'
ARG_RESUME = 'resume'
ARG_AMP = 'amp'
# Train: Optimizer
ARG_CLIP_MAX_NORM = 'clip_max_norm'
ARG_OPTIM_PARAMS = 'optim_params'
ARG_OPTIMIZER = 'optimizer'
ARG_LR_POLICY = 'lr_policy'
# Train: Loss
ARG_L1 = 'l1'
ARG_SEG_CE = 'seg_ce'
ARG_SEG_CE_AUX1 = 'seg_ce_aux1'
ARG_SEG_CE_AUX2 = 'seg_ce_aux2'
# Train: Debug
ARG_DETECT_ANOMALY = 'detect_anomaly'
# Train: Other
ARG_NUM_CLASSES = 'num_classes'
ARG_RANGE_NORM = 'range_norm'
ARG_CAMERA_SHAPE = 'camera_shape'
ARG_PARAMS = 'params'
ARG_DATE = 'date'
# Eval: Evaluate
ARG_EVAL_DATA = 'eval_data'
ARG_EVAL_DL_CONFIG = 'eval_dl_config'
ARG_CHECK_POINT = 'check_point'
ARG_TRAIN_CONFIG = 'train_config'
ARG_THRESHOLDS = 'thresholds'
ARG_NOMAP = 'nomap'
# Eval: Other
ARG_LABEL_TAGS = 'label_tags'
# Optuna
ARG_SAMPLER = 'sampler'
ARG_SEED = 'seed'
ARG_FUNC = 'func'
ARG_N_TRIALS = 'n_trials'
ARG_HOST = 'host'
# TF-Check
ARG_DATA = 'data'
ARG_DL_CONFIG = 'dl_config'
# CKPT2TSM
ARG_HEIGHT = 'height'
ARG_WIDTH = 'width'

DEFAULT_THRESHOLDS = [
    5.0,
    6.0,
    7.0,
    8.0,
    9.0,
    10.0,
    15.0,
    20.0,
    30.0,
    40.0,
    50.0,
    60.0,
    70.0,
    80.0
]

DATASET_CAMERA = 'camera'
DATASET_MAP = 'map'
DATASET_LABEL = 'label'
DATASET_DEPTH = 'depth'
DATASET_POSE_ERR = 'pose_err'

DIR_CHECKPOINTS = 'checkpoints'
DIR_LOGS = 'logs'
DIR_RESULTS = 'results'
DIR_OPTUNA = 'optuna'

SINGLE_SAMPLER_TPE = 'tpe'
SINGLE_SAMPLER_GRID = 'grid'
SINGLE_SAMPLER_RANDOM = 'random'
SINGLE_SAMPLER_CMAES = 'cmaes'
SINGLE_SAMPLER_PARTIALFIXED = 'partialfixed'
SINGLE_SAMPLERS = [
    SINGLE_SAMPLER_TPE,
    SINGLE_SAMPLER_GRID,
    SINGLE_SAMPLER_RANDOM,
    SINGLE_SAMPLER_CMAES,
]

MULTI_SAMPLER_NSGA2 = 'nsga2'
MULTI_SAMPLER_MOTPE = 'motpe'
MULTI_SAMPLERS = [
    MULTI_SAMPLER_NSGA2,
    MULTI_SAMPLER_MOTPE,
]

CONFIG_ENCODER = 'encoder'
CONFIG_DECODER = 'decoder'
CONFIG_DEPTH = 'depth'
CONFIG_SEGMENTATION = 'segmentation'
CONFIG_OPTIMIZER = 'optimizer'
CONFIG_SCHEDULER = 'scheduler'

CONFIG_LR = 'lr'
CONFIG_NITER = 'niter'
CONFIG_BETA1 = 'beta1'
CONFIG_BETA2 = 'beta2'
CONFIG_MOMENTUM = 'momentum'
CONFIG_EPOCH_COUNT = 'epoch_count'
CONFIG_NITER = 'niter'
CONFIG_NITER_DECAY = 'niter_decay'
CONFIG_GAMMA = 'gamma'
CONFIG_PATIENCE = 'patience'
CONFIG_FACTOR = 'factor'
CONFIG_THRESHOLD = 'threshold'
CONFIG_ETA_MIN = 'eta_min'
CONFIG_BASE_LR = 'base_lr'
CONFIG_MAX_LR = 'max_lr'

OPTIM_TYPE_ADAM = 'adam'
OPTIM_TYPE_SGD = 'sgd'
OPTIM_TYPE_ADABELIEF = 'adabelief'
OPTIM_TYPES = [
    OPTIM_TYPE_ADAM,
    OPTIM_TYPE_SGD,
    OPTIM_TYPE_ADABELIEF,
]

LR_POLICY_LAMBDA = 'lambda'
LR_POLICY_STEP = 'step'
LR_POLICY_PLATEAU = 'plateau'
LR_POLICY_COS = 'cos'
LR_POLICY_CLR = 'clr'
LR_POLICIES = [
    LR_POLICY_LAMBDA, LR_POLICY_STEP, LR_POLICY_PLATEAU,
    LR_POLICY_COS, LR_POLICY_CLR,
]

INIT_TYPE_NORMAL = 'normal'
INIT_TYPE_XAVIER = 'xavier'
INIT_TYPE_KAIMING = 'kaiming'
INIT_TYPE_ORTHOGONAL = 'orthogonal'

METRIC_IOU = 'iou'
METRIC_MAE = 'mae'
METRIC_RMSE = 'rmse'
METRIC_MAPE = 'mape'

PADDING_ZEROS = 'zeros'
PADDING_REFLECT = 'reflect'
PADDING_REPLICATE = 'replicate'
PADDING_CIRCULAR = 'circular'

UPSAMPLE_BILINEAR = 'bilinear'
