# PMOD-Net: point-cloud-map-based metric scale obstacle detection by using a monocular camera

[日本語](./README-JP.md)

## Requirement

- NVIDIA-Driver `>=418.81.07`
- Docker `>=19.03`
- NVIDIA-Docker2

## Docker Images

- pull
    ```bash
    docker pull shikishimatasakilab/pmod
    ```

- build
    ```bash
    docker pull pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel
    ```
    ```bash
    ./docker/build.sh -i pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel
    ```

### If you use Optuna
- Pull the Docker image with the following command.
    ```bash
    docker pull mysql
    ```

## Preparing Datasets

### KITTI-360

1. Store the KITTI-360 dataset in HDF5 using "[h5_kitti360](https://github.com/shikishima-TasakiLab/h5_kitti360)".
1. For the dataloader configuration file, use `./config/kitti360-5class.json` for training and `./config/kitti360-5class-ins.json` for validation and evaluation.

### Other Datasets

1. See this [file](OTHER_DATASETS.md).

## Start a Docker Container

1. Start a Docker container with the following command.
    ```bash
    ./docker/run.sh -d path/of/the/dataset/dir
    ```
    ```text
    Usage: run.sh [OPTIONS...]
    OPTIONS:
        -h, --help          Show this help
        -i, --gpu-id ID     Specify the ID of the GPU
        -d, --dataset-dir   Specify the directory where datasets are stored
    ```

## Training

1. Start training with the following command.
    ```bash
    python train.py -t TAG -tdc path/of/the/config.json \
      -td path/of/the/dataset1.hdf5 [path/of/the/dataset1.hdf5 ...] \
      -vdc path/of/the/config.json \
      -bs BLOCK_SIZE
    ```
    ```text
    usage: train.py [-h] -t TAG -tdc PATH [-vdc PATH] [-bs BLOCK_SIZE]
                    -td PATH [PATH ...] [-vd [PATH [PATH ...]]] [--epochs EPOCHS]
                    [--epoch-start-count EPOCH_START_COUNT]
                    [--steps-per-epoch STEPS_PER_EPOCH] [-ppa]
                    [--tr-error-range TR_ERROR_RANGE TR_ERROR_RANGE TR_ERROR_RANGE]
                    [--rot-error-range ROT_ERROR_RANGE] [-ae] [-edc PATH]
                    [-ed [PATH [PATH ...]]] [--thresholds THRESHOLDS [THRESHOLDS ...]]
                    [--seed SEED] [-b BATCH_SIZE] [--resume PATH] [-amp]
                    [--clip-max-norm CLIP_MAX_NORM] [-op PATH]
                    [-o {adam,sgd,adabelief}] [-lp {lambda,step,plateau,cos,clr}]
                    [--l1 L1] [--seg-ce SEG_CE] [--seg-ce-aux1 SEG_CE_AUX1]
                    [--seg-ce-aux2 SEG_CE_AUX2] [--detect-anomaly]

    optional arguments:
      -h, --help            show this help message and exit

    Training:
      -t TAG, --tag TAG     Training Tag.
      -tdc PATH, --train-dl-config PATH
                            PATH of JSON file of dataloader config for training.
      -vdc PATH, --val-dl-config PATH
                            PATH of JSON file of dataloader config for validation.
                            If not specified, the same file as "--train-dl-config" will be used.
      -bs BLOCK_SIZE, --block-size BLOCK_SIZE
                            Block size of dataset.
      -td PATH [PATH ...], --train-data PATH [PATH ...]
                            PATH of training HDF5 datasets.
      -vd [PATH [PATH ...]], --val-data [PATH [PATH ...]]
                            PATH of validation HDF5 datasets. If not specified,
                            the same files as "--train-data" will be used.
      --epochs EPOCHS       Epochs
      --epoch-start-count EPOCH_START_COUNT
                            The starting epoch count
      --steps-per-epoch STEPS_PER_EPOCH
                            Number of steps per epoch. If it is greater than the total number
                            of datasets, then the total number of datasets is used.
      -ppa, --projected-position-augmentation
                            Unuse Projected Positiion Augmentation
      --tr-error-range TR_ERROR_RANGE TR_ERROR_RANGE TR_ERROR_RANGE
                            Translation Error Range [m].
      --rot-error-range ROT_ERROR_RANGE
                            Rotation Error Range [deg].
      -ae, --auto-evaluation
                            Auto Evaluation.
      -edc PATH, --eval-dl-config PATH
                            PATH of JSON file of dataloader config.
      -ed [PATH [PATH ...]], --eval-data [PATH [PATH ...]]
                            PATH of evaluation HDF5 datasets.
      --thresholds THRESHOLDS [THRESHOLDS ...]
                            Thresholds of depth.
      --seed SEED           Random seed.

    Network:
      -b BATCH_SIZE, --batch-size BATCH_SIZE
                            Batch Size
      --resume PATH         PATH of checkpoint(.pth).
      -amp, --amp           Use AMP.

    Optimizer:
      --clip-max-norm CLIP_MAX_NORM
                            max_norm for clip_grad_norm.
      -op PATH, --optim-params PATH
                            PATH of YAML file of optimizer params.
      -o {adam,sgd,adabelief}, --optimizer {adam,sgd,adabelief}
                            Optimizer
      -lp {lambda,step,plateau,cos,clr}, --lr-policy {lambda,step,plateau,cos,clr}
                            Learning rate policy.

    Loss:
      --l1 L1               Weight of L1 loss.
      --seg-ce SEG_CE       Weight of Segmentation CrossEntropy Loss.
      --seg-ce-aux1 SEG_CE_AUX1
                            Weight of Segmentation Aux1 CrosEntropy Loss.
      --seg-ce-aux2 SEG_CE_AUX2
                            Weight of Segmentation Aux2 CrosEntropy Loss.

    Debug:
      --detect-anomaly      AnomalyMode
    ```

1. The checkpoints of the training will be stored in the "./checkpoints" directory.

    ```text
    checkpoints/
    　├ YYYYMMDDThhmmss-TAG/
    　│　├ config.yaml
    　│　├ 00001_PMOD.pth
    　│　├ :
    　│　├ :
    　│　├ EPOCH_PMOD.pth
    　│　└ validation.xlsx
    ```

## Evaluation

1. Start evaluation with the following command.
    ```bash
    python evaluate.py -t TAG -cp path/of/the/checkpoint.pth \
      -edc path/of/the/config.json \
      -ed path/of/the/dataset1.hdf5 [path/of/the/dataset2.hdf5 ...]
    ```
    ```text
    usage: evaluate.py [-h] -t TAG -cp PATH -edc PATH [-bs BLOCK_SIZE]
                       -ed PATH [PATH ...] [--train-config PATH]
                       [--thresholds THRESHOLDS [THRESHOLDS ...]]
                       [--seed SEED] [-b BATCH_SIZE]

    optional arguments:
      -h, --help            show this help message and exit

    Evaluation:
      -t TAG, --tag TAG     Evaluation Tag.
      -cp PATH, --check-point PATH
                            PATH of checkpoint.
      -edc PATH, --eval-dl-config PATH
                            PATH of JSON file of dataloader config.
      -bs BLOCK_SIZE, --block-size BLOCK_SIZE
                            Block size of dataset.
      -ed PATH [PATH ...], --eval-data PATH [PATH ...]
                            PATH of evaluation HDF5 datasets.
      --nomap               No map input.
      --tr-error-range TR_ERROR_RANGE TR_ERROR_RANGE TR_ERROR_RANGE
                            Translation Error Range [m]. This is used when the
                            data do not contain poses.
      --rot-error-range ROT_ERROR_RANGE
                            Rotation Error Range [deg]. This is used when the data
                            do not contain poses.
      --train-config PATH   PATH of "config.yaml"
      --thresholds THRESHOLDS [THRESHOLDS ...]
                            Thresholds of depth.
      --seed SEED           Random seed.

    Network:
      -b BATCH_SIZE, --batch-size BATCH_SIZE
                            Batch Size
    ```

1. The results of the evaluation will be stored in the "./results" directory.

    ```text
    results/
    　├ YYYYMMDDThhmmss-TRAINTAG/
    　│　├ YYYYMMDDThhmmss-TAG/
    　│　│　├ config.yaml
    　│　│　├ data.hdf5
    　│　│　└ result.xlsx
    ```

## data.hdf5 &rarr; Video (.avi)

1. Convert "data.hdf5" to video (.avi) with the following command.
    ```bash
    python data2avi.py -i path/of/the/data.hdf5
    ```
    ```text
    usage: data2avi.py [-h] -i PATH [-o PATH] [-r]

    optional arguments:
      -h, --help            show this help message and exit
      -i PATH, --input PATH
                            Input path.
      -o PATH, --output PATH
                            Output path. Default is "[input dir]/data.avi"
      -r, --raw             Use raw codec
    ```

1. The converted video will be output to the same directory as the input HDF5 file.

## Checkpoint (.pth) &rarr; Torch Script model (.pt)

1. Convert checkpoint (.pth) to Torch Script model (.pt) with the following command.
    ```bash
    python ckpt2tsm.py -c path/of/the/checkpoint.pth
    ```
    ```text
    usage: ckpt2tsm.py [-h] [-c PATH] [-x WIDTH] [-y HEIGHT]

    optional arguments:
      -h, --help            show this help message and exit
      -c PATH, --check-point PATH
                            Path of checkpoint file.
      -x WIDTH, --width WIDTH
                            Width of input images.
      -y HEIGHT, --height HEIGHT
                            Height of input images.
    ```

1. The converted Torch Script model will be output to the same directory as the input checkpoint.

## Training with Optuna

1. Start a Docker container for MySQL with the following command in another terminal.
    ```bash
    ./optuna/run-mysql.sh
    ```

1. Start training with the following command.
    ```bash
    python optuna_train.py -t TAG -tdc path/of/the/config.json \
      -td path/of/the/dataset1.hdf5 [path/of/the/dataset2.hdf5 ...] -bs BLOCK_SIZE
    ```
    ```text
    usage: optuna_train.py [-h] [--seed SEED] [-n N_TRIALS] -t TAG [-H HOST]
                           [-s {tpe,grid,random,cmaes}] -tdc PATH [-vdc PATH]
                           [-bs BLOCK_SIZE] -td PATH [PATH ...] [-vd [PATH [PATH ...]]]
                           [--epochs EPOCHS] [--epoch-start-count EPOCH_START_COUNT]
                           [--steps-per-epoch STEPS_PER_EPOCH] [-ppa]
                          [--tr-error-range TR_ERROR_RANGE]
                          [--rot-error-range ROT_ERROR_RANGE]
                          [-b BATCH_SIZE] [--resume PATH] [-amp]
                          [--clip-max-norm CLIP_MAX_NORM] [-op PATH]
                          [-o {adam,sgd,adabelief}] [-lp {lambda,step,plateau,cos,clr}]
                          [--l1 L1] [--seg-ce SEG_CE] [--seg-ce-aux1 SEG_CE_AUX1]
                          [--seg-ce-aux2 SEG_CE_AUX2] {multi} ...

    positional arguments:
      {multi}
        multi               Multi Objective Trial

    optional arguments:
      -h, --help            show this help message and exit

    Optuna:
      --seed SEED           Seed for random number generator.
      -n N_TRIALS, --n-trials N_TRIALS
                            Number of trials.
      -t TAG, --tag TAG     Optuna training tag.
      -H HOST, --host HOST  When using a MySQL server, specify the hostname.
      -s {tpe,grid,random,cmaes}, --sampler {tpe,grid,random,cmaes}
                            Optuna sampler.

    Training:
      -tdc PATH, --train-dl-config PATH
                            PATH of JSON file of dataloader config for training.
      -vdc PATH, --val-dl-config PATH
                            PATH of JSON file of dataloader config for validation.
                            If not specified, the same file as "--train-dl-config"
                            will be used.
      -bs BLOCK_SIZE, --block-size BLOCK_SIZE
                            Block size of dataset.
      -td PATH [PATH ...], --train-data PATH [PATH ...]
                            PATH of training HDF5 datasets.
      -vd [PATH [PATH ...]], --val-data [PATH [PATH ...]]
                            PATH of validation HDF5 datasets. If not specified,
                            the same files as "--train-data" will be used.
      --epochs EPOCHS       Epochs
      --epoch-start-count EPOCH_START_COUNT
                            The starting epoch count
      --steps-per-epoch STEPS_PER_EPOCH
                            Number of steps per epoch. If it is greater than the
                            total number of datasets, then the total number of
                            datasets is used.
      -ppa, --projected-position-augmentation
                            Unuse Projected Positiion Augmentation
      --tr-error-range TR_ERROR_RANGE
                            Translation Error Range [m].
      --rot-error-range ROT_ERROR_RANGE
                            Rotation Error Range [deg].

    Network:
      -b BATCH_SIZE, --batch-size BATCH_SIZE
                            Batch Size
      --resume PATH         PATH of checkpoint(.pth).
      -amp, --amp           Use AMP.

    Optimizer:
      --clip-max-norm CLIP_MAX_NORM
                            max_norm for clip_grad_norm.
      -op PATH, --optim-params PATH
                            PATH of YAML file of optimizer params.
      -o {adam,sgd,adabelief}, --optimizer {adam,sgd,adabelief}
                            Optimizer
      -lp {lambda,step,plateau,cos,clr}, --lr-policy {lambda,step,plateau,cos,clr}
                            Learning rate policy.

    Loss:
      --l1 L1               Weight of L1 loss.
      --seg-ce SEG_CE       Weight of Segmentation CrossEntropy Loss.
      --seg-ce-aux1 SEG_CE_AUX1
                            Weight of Segmentation Aux1 CrosEntropy Loss.
      --seg-ce-aux2 SEG_CE_AUX2
                            Weight of Segmentation Aux2 CrosEntropy Loss.
    ```

1. To run parallel training on other machines, specify the server with the "-H" option.

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{pmodnet,
    author = {Junya Shikishima and Keisuke Urasaki and Tsuyoshi Tasaki},
    title = {PMOD-Net: point-cloud-map-based metric scale obstacle detection by using a monocular camera},
    journal = {Advanced Robotics},
    volume = {37},
    number = {7},
    pages = {458-466},
    year  = {2023},
    publisher = {Taylor & Francis},
    doi = {10.1080/01691864.2022.2153080},
    URL = {https://doi.org/10.1080/01691864.2022.2153080},
    eprint = {https://doi.org/10.1080/01691864.2022.2153080}
}
```
