import os
from pmod.train import main, parse_args

if __name__ == '__main__':
    workdir = os.path.dirname(os.path.abspath(__file__))
    args = parse_args()
    main(args, workdir)
