import os
from pmod.evaluate import main, parse_args
import numpy as np
import torch
from pmod.model.constant import ARG_SEED

if __name__ == '__main__':
    workdir = os.path.dirname(os.path.abspath(__file__))
    args = parse_args()

    ###############
    # Random Seed #
    ###############
    np.random.seed(seed=args[ARG_SEED])
    torch.random.manual_seed(seed=args[ARG_SEED])

    main(args, workdir)
