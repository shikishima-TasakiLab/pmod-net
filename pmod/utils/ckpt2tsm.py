import argparse
from typing import Dict
import os

import torch
from torch import Tensor
import torch.nn as nn

from pmod.model.constant import ARG_CHECK_POINT, ARG_HEIGHT, ARG_WIDTH, arg_hyphen


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', f'--{arg_hyphen(ARG_CHECK_POINT)}', type=str, metavar='PATH',
        help='Path of checkpoint file.'
    )
    parser.add_argument(
        '-x', f'--{arg_hyphen(ARG_WIDTH)}', type=int, default=512,
        help='Width of input images.'
    )
    parser.add_argument(
        '-y', f'--{arg_hyphen(ARG_HEIGHT)}', type=int, default=256,
        help='Height of input images.'
    )
    return vars(parser.parse_args())


def main(args: Dict[str, str]):
    netG: nn.Module = torch.load(args[ARG_CHECK_POINT])
    netG.eval()
    netG.cpu()

    # , device=torch.device('cuda'))
    sample_rgb: Tensor = torch.empty(1, 3, args[ARG_HEIGHT], args[ARG_WIDTH])
    # , device=torch.device('cuda'))
    sample_depth: Tensor = torch.empty(1, 1, args[ARG_HEIGHT], args[ARG_WIDTH])

    traced_ts = torch.jit.trace(netG, (sample_rgb, sample_depth))
    traced_ts.save(f'{os.path.splitext(args[ARG_CHECK_POINT])[0]}.pt')
