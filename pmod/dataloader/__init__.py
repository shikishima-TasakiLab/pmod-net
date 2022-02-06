from typing import Dict, List, Tuple, Union
from h5dataloader.pytorch.hdf5dataset import HDF5Dataset
from torch.utils.data.dataloader import DataLoader
from h5dataloader.common.structure import CONFIG_TAG_LABEL, CONFIG_TAG_LABELTAG, CONFIG_TAG_RANGE, CONFIG_TAG_SHAPE, CONFIG_TAG_TAG
from ..model.constant import *
from .pmod_dataset import PMOD_Train_Dataset, PMOD_Test_Dataset


def init_train_dataloader(args: Dict[str, Union[int, float, str, dict]]) -> DataLoader:
    train_use_mods: Tuple[int, int] = None
    block_size: int = args[ARG_BLOCK_SIZE]

    if block_size > 2:
        train_use_mods = (0, block_size - 2)

    tr_err_range = args[ARG_TR_ERROR_RANGE] if args[ARG_PROJECTED_POSITION_AUGMENTATION] is True else 0.0
    rot_err_range = args[ARG_ROT_ERROR_RANGE] if args[ARG_PROJECTED_POSITION_AUGMENTATION] is True else 0.0

    train_dataset = PMOD_Train_Dataset(
        h5_paths=args[ARG_TRAIN_DATA], config=args[ARG_TRAIN_DL_CONFIG], quiet=True,
        block_size=block_size, use_mods=train_use_mods,
        tr_err_range=tr_err_range, rot_err_range=rot_err_range
    )

    camera_shape: Tuple[int, int,
                        int] = train_dataset.minibatch[DATASET_CAMERA][CONFIG_TAG_SHAPE]
    args[ARG_CAMERA_SHAPE] = [camera_shape[2], camera_shape[0], camera_shape[1]]
    args[ARG_NUM_CLASSES] = len(
        train_dataset.label_color_configs[train_dataset.minibatch[DATASET_LABEL][CONFIG_TAG_LABELTAG]])
    args[ARG_RANGE_NORM] = train_dataset.minibatch[DATASET_DEPTH][CONFIG_TAG_RANGE][1]

    return DataLoader(train_dataset, batch_size=args[ARG_BATCH_SIZE], shuffle=True)


def init_val_dataloader(args: Dict[str, Union[int, float, str, dict]]) -> DataLoader:
    val_use_mods: Tuple[int, int] = None
    block_size: int = args[ARG_BLOCK_SIZE]

    if block_size > 2:
        val_use_mods = (block_size - 2, block_size - 1)

    val_dataset = PMOD_Test_Dataset(
        h5_paths=args[ARG_VAL_DATA], config=args[ARG_VAL_DL_CONFIG], quiet=True,
        block_size=block_size, use_mods=val_use_mods,
        tr_err_range=args.get(ARG_TR_ERROR_RANGE), rot_err_range=args.get(ARG_ROT_ERROR_RANGE)
    )
    return DataLoader(val_dataset, batch_size=args[ARG_BATCH_SIZE], shuffle=False)


def init_eval_dataloader(args: Dict[str, Union[int, float, str, dict]]) -> DataLoader:
    eval_use_mods: Tuple[int, int] = None
    block_size: int = args.get(ARG_BLOCK_SIZE, 0)

    if block_size > 2:
        eval_use_mods = (block_size - 1, block_size)

    eval_dataset = PMOD_Test_Dataset(
        h5_paths=args[ARG_EVAL_DATA], config=args[ARG_EVAL_DL_CONFIG], quiet=True,
        block_size=block_size, use_mods=eval_use_mods,
        tr_err_range=args.get(ARG_TR_ERROR_RANGE), rot_err_range=args.get(ARG_ROT_ERROR_RANGE)
    )

    label_color_configs: List[Dict[str, Union[str, int]]
                              ] = eval_dataset.label_color_configs[eval_dataset.minibatch[DATASET_LABEL][CONFIG_TAG_LABELTAG]]

    args[ARG_LABEL_TAGS] = {
        str(lcc[CONFIG_TAG_LABEL]): lcc[CONFIG_TAG_TAG] for lcc in label_color_configs}
    args[ARG_NUM_CLASSES] = len(label_color_configs)
    args[ARG_RANGE_NORM] = eval_dataset.minibatch[DATASET_DEPTH][CONFIG_TAG_RANGE][1]

    return DataLoader(eval_dataset, batch_size=args[ARG_BATCH_SIZE], shuffle=False)


def init_tfcheck_dataloader(args: Dict[str, Union[int, float, str, dict]]) -> DataLoader:
    tfcheck_dataset = HDF5Dataset(
        h5_paths=args[ARG_DATA], config=args[ARG_DL_CONFIG], quiet=True)
    return DataLoader(tfcheck_dataset, batch_size=None, shuffle=False)
