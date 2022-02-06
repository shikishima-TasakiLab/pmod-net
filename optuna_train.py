from pmod.optuna_train import *

if __name__ == '__main__':
    workdir = os.path.dirname(os.path.abspath(__file__))
    args = parse_args()
    args[ARG_FUNC](args, workdir)
