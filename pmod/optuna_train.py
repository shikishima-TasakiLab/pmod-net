import argparse
import codecs
from typing import Dict, Union
import os
import yaml
import optuna
from .model.constant import *
from .train import main as train

DB_URL = 'mysql+pymysql://pmod:pmod@{host}:13306/optuna_pmod?charset=utf8'
RESULT_DIR = os.path.join(DIR_OPTUNA, DIR_RESULTS)

def parse_args() -> Dict[str, str]:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_optuna = parser.add_argument_group('Optuna')
    parser_optuna.add_argument(
        f'--{arg_hyphen(ARG_SEED)}',
        type=int, default=1,
        help='Seed for random number generator.'
    )
    parser_optuna.add_argument(
        '-n', f'--{arg_hyphen(ARG_N_TRIALS)}',
        type=int, default=200,
        help='Number of trials.'
    )
    parser_optuna.add_argument(
        '-t', f'--{arg_hyphen(ARG_TAG)}',
        type=str, required=True,
        help='Optuna training tag.'
    )
    parser_optuna.add_argument(
        '-H', f'--{arg_hyphen(ARG_HOST)}',
        type=str, default=None,
        help='When using a MySQL server, specify the hostname.'
    )
    parser_optuna.add_argument(
        '-s', f'--{arg_hyphen(ARG_SAMPLER)}',
        choices=SINGLE_SAMPLERS, default=SINGLE_SAMPLER_TPE,
        help='Optuna sampler.'
    )
    parser.set_defaults(func=single_main)

    parser_train = parser.add_argument_group('Training')
    parser_train.add_argument(
        '-tdc', f'--{arg_hyphen(ARG_TRAIN_DL_CONFIG)}',
        type=str, metavar='PATH', required=True,
        help='PATH of JSON file of dataloader config for training.'
    )
    parser_train.add_argument(
        '-vdc', f'--{arg_hyphen(ARG_VAL_DL_CONFIG)}',
        type=str, metavar='PATH', default=None,
        help=f'PATH of JSON file of dataloader config for validation. If not specified, the same file as "--{arg_hyphen(ARG_TRAIN_DL_CONFIG)}" will be used.'
    )
    parser_train.add_argument(
        '-bs', f'--{arg_hyphen(ARG_BLOCK_SIZE)}',
        type=int, default=0,
        help='Block size of dataset.'
    )
    parser_train.add_argument(
        '-td', f'--{arg_hyphen(ARG_TRAIN_DATA)}',
        type=str, metavar='PATH', nargs='+', required=True,
        help='PATH of training HDF5 datasets.'
    )
    parser_train.add_argument(
        '-vd', f'--{arg_hyphen(ARG_VAL_DATA)}',
        type=str, metavar='PATH', nargs='*', default=[],
        help=f'PATH of validation HDF5 datasets. If not specified, the same files as "--{arg_hyphen(ARG_TRAIN_DATA)}" will be used.'
    )
    parser_train.add_argument(
        f'--{arg_hyphen(ARG_EPOCHS)}',
        type=int, default=50,
        help='Epochs'
    )
    parser_train.add_argument(
        f'--{arg_hyphen(ARG_EPOCH_START_COUNT)}',
        type=int, default=1,
        help='The starting epoch count'
    )
    parser_train.add_argument(
        f'--{arg_hyphen(ARG_STEPS_PER_EPOCH)}',
        type=int, default=10000,
        help='Number of steps per epoch. If it is greater than the total number of datasets, then the total number of datasets is used.'
    )
    parser_train.add_argument(
        '-ppa', f'--{arg_hyphen(ARG_PROJECTED_POSITION_AUGMENTATION)}',
        action='store_false', help='Unuse Projected Positiion Augmentation'
    )
    parser_train.add_argument(
        f'--{arg_hyphen(ARG_TR_ERROR_RANGE)}',
        type=float, default=[0.6, 1.3, 0.7],
        help='Translation Error Range [m].'
    )
    parser_train.add_argument(
        f'--{arg_hyphen(ARG_ROT_ERROR_RANGE)}',
        type=float, default=3.0,
        help='Rotation Error Range [deg].'
    )

    parser_net = parser.add_argument_group('Network')
    parser_net.add_argument(
        '-b', f'--{arg_hyphen(ARG_BATCH_SIZE)}',
        type=int, default=2,
        help='Batch Size'
    )
    parser_net.add_argument(
        f'--{arg_hyphen(ARG_RESUME)}',
        type=str, metavar='PATH', default=None,
        help='PATH of checkpoint(.pth).'
    )
    parser_net.add_argument(
        f'-amp', f'--{arg_hyphen(ARG_AMP)}',
        action='store_true',
        help='Use AMP.'
    )

    parser_optim = parser.add_argument_group('Optimizer')
    parser_optim.add_argument(
        f'--{arg_hyphen(ARG_CLIP_MAX_NORM)}',
        type=float, default=1.0,
        help='max_norm for clip_grad_norm.'
    )
    parser_optim.add_argument(
        '-op', f'--{arg_hyphen(ARG_OPTIM_PARAMS)}',
        type=str, metavar='PATH', default='./config/optim-params-default.yaml', help='PATH of YAML file of optimizer params.'
    )
    parser_optim.add_argument(
        '-o', f'--{arg_hyphen(ARG_OPTIMIZER)}',
        type=str, default=None, choices=OPTIM_TYPES,
        help='Optimizer'
    )
    parser_optim.add_argument(
        '-lp', f'--{arg_hyphen(ARG_LR_POLICY)}',
        type=str, default=LR_POLICY_PLATEAU, choices=LR_POLICIES,
        help='Learning rate policy.'
    )

    parser_loss = parser.add_argument_group('Loss')
    parser_loss.add_argument(
        f'--{arg_hyphen(ARG_L1)}',
        type=float, default=None,
        help='Weight of L1 loss.'
    )
    parser_loss.add_argument(
        f'--{arg_hyphen(ARG_SEG_CE)}',
        type=float, default=None,
        help='Weight of Segmentation CrossEntropy Loss.'
    )
    parser_loss.add_argument(
        f'--{arg_hyphen(ARG_SEG_CE_AUX1)}',
        type=float, default=None,
        help='Weight of Segmentation Aux1 CrosEntropy Loss.'
    )
    parser_loss.add_argument(
        f'--{arg_hyphen(ARG_SEG_CE_AUX2)}',
        type=float, default=None,
        help='Weight of Segmentation Aux2 CrosEntropy Loss.'
    )

    parser_multi = subparsers.add_parser(
        'multi', help='Multi Objective Trial'
    )
    parser_multi.add_argument(
        '-s', f'--{arg_hyphen(ARG_SAMPLER)}',
        choices=MULTI_SAMPLERS, default=MULTI_SAMPLER_MOTPE,
        help='Optuna sampler.'
    )
    parser_multi.set_defaults(func=multi_main)

    args = vars(parser.parse_args())

    if os.path.isfile(args[ARG_OPTIM_PARAMS]) is False:
        raise FileNotFoundError(f'"{args[ARG_OPTIM_PARAMS]}"')
    with open(args[ARG_OPTIM_PARAMS]) as f:
        optim_params:dict = yaml.safe_load(f)
    args[ARG_OPTIM_PARAMS] = optim_params

    args[ARG_EVAL_DATA] = []
    args[ARG_DETECT_ANOMALY] = False

    return args

def objective_with_args(args: Dict[str, str], workdir: str):

    def objective(trial: optuna.Trial):
        train_args: Dict[str, str] = args.copy()
        train_args.pop('func', None)
        trial.set_user_attr('hostname', os.environ.get('HOST_NAME', os.environ['HOSTNAME']))

        train_args[ARG_TAG] = f'optuna-trial{trial.number:06d}-{args[ARG_TAG]}'

        # Optimizer
        train_args[ARG_OPTIMIZER] = trial.suggest_categorical(ARG_OPTIMIZER, OPTIM_TYPES) if args[ARG_OPTIMIZER] is None else args[ARG_OPTIMIZER]

        optim_params: Dict[str, Dict[str, Dict[str, Dict[str, Union[int, float]]]]] = train_args[ARG_OPTIM_PARAMS]
        for module_key, module_dict in optim_params.items():
            optim_dict: Dict[str, Union[float, Dict[str, float]]] = module_dict[CONFIG_OPTIMIZER]
            if optim_dict.get(CONFIG_LR) is None:
                optim_dict[CONFIG_LR] = trial.suggest_loguniform(CONFIG_LR, 1e-9, 1e-1)
            if train_args[ARG_OPTIMIZER] in [OPTIM_TYPE_ADAM, OPTIM_TYPE_ADABELIEF]:
                optim_param_dict: Dict[str, float] = optim_dict[train_args[ARG_OPTIMIZER]]
                if optim_param_dict.get(CONFIG_BETA1) is None:
                    optim_param_dict[CONFIG_BETA1] = trial.suggest_uniform(CONFIG_BETA1, 0.0, 1.0)
                if optim_param_dict.get(CONFIG_BETA2) is None:
                    optim_param_dict[CONFIG_BETA2] = trial.suggest_uniform(CONFIG_BETA2, 0.0, 1.0)
            elif train_args[ARG_OPTIMIZER] in [OPTIM_TYPE_SGD]:
                optim_param_dict: Dict[str, float] = optim_dict[train_args[ARG_OPTIMIZER]]
                if optim_param_dict.get(CONFIG_MOMENTUM) is None:
                    optim_param_dict[CONFIG_MOMENTUM] = trial.suggest_float(CONFIG_MOMENTUM, 0.0, 1.0)
            else:
                raise NotImplementedError(train_args[ARG_OPTIMIZER])

            scheduler_dict: Dict[str, Dict[str, Union[int, float]]] = module_dict[CONFIG_SCHEDULER]
            if train_args[ARG_LR_POLICY] in [LR_POLICY_LAMBDA]:
                scheduler_param_dict: Dict[str, Union[int, float]] = scheduler_dict[train_args[ARG_LR_POLICY]]
                if scheduler_param_dict.get(CONFIG_EPOCH_COUNT) is None:
                    scheduler_param_dict[CONFIG_EPOCH_COUNT] = args[ARG_EPOCH_START_COUNT]
                if scheduler_param_dict.get(CONFIG_NITER) is None:
                    scheduler_param_dict[CONFIG_NITER] = trial.suggest_int(CONFIG_NITER, 1, args[ARG_STEPS_PER_EPOCH])
                if scheduler_param_dict.get(CONFIG_NITER_DECAY) is None:
                    scheduler_param_dict[CONFIG_NITER_DECAY] = trial.suggest_int(CONFIG_NITER_DECAY, 1, args[ARG_STEPS_PER_EPOCH])
            elif train_args[ARG_LR_POLICY] in [LR_POLICY_STEP]:
                scheduler_param_dict: Dict[str, Union[int, float]] = scheduler_dict[train_args[ARG_LR_POLICY]]
                if scheduler_param_dict.get(CONFIG_NITER_DECAY) is None:
                    scheduler_param_dict[CONFIG_NITER_DECAY] = trial.suggest_int(CONFIG_NITER_DECAY, low=1)
                if scheduler_param_dict.get(CONFIG_GAMMA) is None:
                    scheduler_param_dict[CONFIG_GAMMA] = trial.suggest_uniform(CONFIG_GAMMA, 1e-3, 9e-1)
            elif train_args[ARG_LR_POLICY] in [LR_POLICY_PLATEAU]:
                scheduler_param_dict: Dict[str, Union[int, float]] = scheduler_dict[train_args[ARG_LR_POLICY]]
                if scheduler_param_dict.get(CONFIG_PATIENCE) is None:
                    scheduler_param_dict[CONFIG_PATIENCE] = trial.suggest_int(CONFIG_PATIENCE, 1, 100)
                if scheduler_param_dict.get(CONFIG_FACTOR) is None:
                    scheduler_param_dict[CONFIG_FACTOR] = trial.suggest_uniform(CONFIG_FACTOR, 1e-3, 9e-1)
                if scheduler_param_dict.get(CONFIG_THRESHOLD) is None:
                    scheduler_param_dict[CONFIG_THRESHOLD] = trial.suggest_uniform(CONFIG_THRESHOLD, 1e-9, 1e-1)
            elif train_args[ARG_LR_POLICY] in [LR_POLICY_COS]:
                scheduler_param_dict: Dict[str, Union[int, float]] = scheduler_dict[train_args[ARG_LR_POLICY]]
                if scheduler_param_dict.get(CONFIG_NITER) is None:
                    scheduler_param_dict[CONFIG_NITER] = trial.suggest_int(CONFIG_NITER, 1, args[ARG_STEPS_PER_EPOCH])
                if scheduler_param_dict.get(CONFIG_ETA_MIN) is None:
                    scheduler_param_dict[CONFIG_ETA_MIN] = trial.suggest_uniform(CONFIG_ETA_MIN, 1e-9, 1e-1)
            elif train_args[ARG_LR_POLICY] in [LR_POLICY_CLR]:
                scheduler_param_dict: Dict[str, Union[int, float]] = scheduler_dict[train_args[ARG_LR_POLICY]]
                if scheduler_param_dict.get(CONFIG_BASE_LR) is None:
                    scheduler_param_dict[CONFIG_BASE_LR] = trial.suggest_uniform(CONFIG_BASE_LR, 1e-9, 1e-1)
                if scheduler_param_dict.get(CONFIG_MAX_LR) is None:
                    scheduler_param_dict[CONFIG_MAX_LR] = trial.suggest_uniform(CONFIG_MAX_LR, 1e-9, 1e-1)
            else:
                raise NotImplementedError(train_args[ARG_LR_POLICY])

        # Loss
        train_args[ARG_L1] = trial.suggest_uniform(ARG_L1, 0.0, 1.0) if args[ARG_L1] is None else args[ARG_L1]
        train_args[ARG_SEG_CE] = trial.suggest_uniform(ARG_SEG_CE, 0.0, 1.0) if args[ARG_SEG_CE] is None else args[ARG_SEG_CE]
        train_args[ARG_SEG_CE_AUX1] = trial.suggest_uniform(ARG_SEG_CE_AUX1, 0.0, 1.0) if args[ARG_SEG_CE_AUX1] is None else args[ARG_SEG_CE_AUX1]
        train_args[ARG_SEG_CE_AUX2] = trial.suggest_uniform(ARG_SEG_CE_AUX2, 0.0, 1.0) if args[ARG_SEG_CE_AUX2] is None else args[ARG_SEG_CE_AUX2]

        print(f'{"Trial":11s}: {trial.number:14d}')
        result: Dict[str, float] = train(train_args, workdir, trial)

        if len(trial.study.directions) > 1:
            return result[METRIC_IOU], result[METRIC_MAPE]
        else:
            return 1.0 - result[METRIC_IOU] + result[METRIC_MAPE]

    return objective

def optimize(study: optuna.Study, args: Dict[str, Union[int, str]], workdir: str, storage: str):
    print(f'{"Optuna":11s}: {"Tag":14s}: {args[ARG_TAG]}')
    print(f'{"":11s}: {"Study":14s}: {study.study_name}')
    print(f'{"":11s}: {"Num Trials":14s}: {args[ARG_N_TRIALS]}')
    print(f'{"":11s}: {"Storage":14s}: {storage}')

    try:
        study.optimize(objective_with_args(args, workdir), n_trials=args[ARG_N_TRIALS])
    except:
        pass
    finally:
        save_path: str = os.path.join(
            workdir,
            RESULT_DIR
        )
        os.makedirs(save_path, exist_ok=True)
        for best_trial in study.best_trials:
            save_path = os.path.join(save_path, f'{best_trial.datetime_start.strftime("%Y%m%dT%H%M%S")}-{args[ARG_TAG]}.yaml')
            with codecs.open(save_path, mode='w', encoding='utf-8') as f:
                yaml.dump(best_trial.params, f, encoding='utf-8', allow_unicode=True)

def single_main(args: Dict[str, str], workdir: str):
    if args[ARG_SAMPLER] == SINGLE_SAMPLER_GRID:
        sampler = optuna.samplers.GridSampler()
    elif args[ARG_SAMPLER] == SINGLE_SAMPLER_RANDOM:
        sampler = optuna.samplers.RandomSampler(seed=args[ARG_SEED])
    elif args[ARG_SAMPLER] == SINGLE_SAMPLER_CMAES:
        sampler = optuna.samplers.CmaEsSampler(seed=args[ARG_SEED])
    else:
        sampler = optuna.samplers.TPESampler(seed=args[ARG_SEED])

    storage: str = DB_URL.replace('{host}', args[ARG_HOST]) if isinstance(args[ARG_HOST], str) else None
    study_name: str = f'pmod-{args[ARG_TAG]}'

    study: optuna.Study = optuna.create_study(
        storage=storage,
        sampler=sampler,
        direction='minimize',
        study_name=study_name,
        load_if_exists=True,
    )
    if 'Objective' not in study.user_attrs.keys():
        study.set_user_attr('Objective', '(1 - mIoU) + MAPE')

    optimize(study, args, workdir, storage)

def multi_main(args: Dict[str, str], workdir: str):
    if args[ARG_SAMPLER] == MULTI_SAMPLER_NSGA2:
        sampler = optuna.samplers.NSGAIISampler(seed=args[ARG_SEED])
    else:
        sampler = optuna.samplers.MOTPESampler(seed=args[ARG_SEED])

    storage: str = DB_URL.replace('{host}', args[ARG_HOST]) if isinstance(args[ARG_HOST], str) else None
    study_name: str = f'pmod-{args[ARG_TAG]}'

    study: optuna.Study = optuna.create_study(
        storage=storage,
        directions=['maximize', 'minimize'],
        sampler=sampler,
        study_name=study_name,
        load_if_exists=True,
    )
    if 'Objectives' not in study.user_attrs.keys():
        study.set_user_attr('Objectives', ['Best mIoU', 'Best MAPE'])

    optimize(study, args, workdir, storage)
