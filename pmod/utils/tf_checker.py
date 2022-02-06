import argparse
from torch import Tensor
import numpy as np
from tqdm import tqdm
import pandas as pd
from ..dataloader import init_tfcheck_dataloader
from ..model.constant import *


def parse_args() -> dict:
    parser = argparse.ArgumentParser()
    parser_eval = parser.add_argument_group('Evaluation')
    parser_eval.add_argument(
        '-dc', f'--{arg_hyphen(ARG_DL_CONFIG)}',
        type=str, metavar='PATH', required=True,
        help='PATH of JSON file of dataloader config.'
    )
    parser_eval.add_argument(
        '-d', f'--{arg_hyphen(ARG_DATA)}',
        type=str, metavar='PATH', nargs='+', required=True,
        help='PATH of evaluation HDF5 datasets.'
    )

    args: dict = vars(parser.parse_args())

    return args


def main(args: dict):
    src_dataloader = init_tfcheck_dataloader(args=args)

    tr_list: np.ndarray = np.empty([0, 3], np.float32)
    q_list: np.ndarray = np.empty([0, 4], np.float32)
    dst_err_list: np.ndarray = np.empty(0, np.float32)
    rot_err_list: np.ndarray = np.empty(0, np.float32)

    for batch in tqdm(src_dataloader):
        tr: Tensor = batch[DATASET_POSE_ERR][0:3]
        q: Tensor = batch[DATASET_POSE_ERR][3:7]

        dst_err: Tensor = torch.linalg.norm(tr)
        rot_err: Tensor = torch.atan2(
            torch.linalg.norm(q[:3]), torch.abs(q[Q_IDX.w])) * 2.0
        rot_err: Tensor = torch.abs(rot_err)

        tr_list = np.append(tr_list, [tr.numpy()], axis=0)
        q_list = np.append(q_list, [q.numpy()], axis=0)
        dst_err_list = np.append(dst_err_list, dst_err)
        rot_err_list = np.append(rot_err_list, np.degrees(rot_err))

    print(f'|{"Metric":14s}|{"Distance [m]":14s}|{"Rotation [deg]":14s}|')
    print(f'|{"":-<14s}|{"":-<14s}|{"":-<14s}|')

    dst_err_min = dst_err_list.min()
    rot_err_min = rot_err_list.min()
    print(f'|{"Min":14s}|{dst_err_min:14f}|{rot_err_min:14f}|')

    dst_err_max = dst_err_list.max()
    rot_err_max = rot_err_list.max()
    print(f'|{"Max":14s}|{dst_err_max:14f}|{rot_err_max:14f}|')

    dst_err_avg = dst_err_list.mean()
    rot_err_avg = rot_err_list.mean()
    print(f'|{"Average":14s}|{dst_err_avg:14f}|{rot_err_avg:14f}|')

    dst_err_mdn = np.median(dst_err_list)
    rot_err_mdn = np.median(rot_err_list)
    print(f'|{"Median":14s}|{dst_err_mdn:14f}|{rot_err_mdn:14f}|')

    dst_err_sstd = np.std(dst_err_list, ddof=1)
    rot_err_sstd = np.std(rot_err_list, ddof=1)
    print(f'|{"Sample STD":14s}|{dst_err_sstd:14f}|{rot_err_sstd:14f}|')

    df = pd.DataFrame(np.concatenate([
        tr_list,
        q_list,
    ], axis=1), columns=[
        'tx [m]', 'ty [m]', 'tz [m]',
        'qx', 'qy', 'qz', 'qw',
    ])
    df.to_excel('tf_checker.xlsx', sheet_name='tf_checker')
