# Other Datasets

1. Create datasets using [this library](https://github.com/shikishima-TasakiLab/h5datacreator).
1. Create a configuration file for the dataloader using [this software](https://github.com/shikishima-TasakiLab/h5dataloader-config).
    
    - **camera**
      - type: *bgr8*
      - shape: `256, 512, 3`
      - normalize: `true`
      - frame-id: The camera coordinate system.
    - **map**
      - type: *depth*
      - shape: `256, 512`
      - normalize: `true`
      - range: `0, 80`
      - frame-id: The camera coordinate system, including the self-localization error if **pose_err** is used.
      - from:
        - *points*, *semantic3d*, *voxel-points*, or *voxel-semantic3d*
        - *intrinsic*: Intrinsic parameters of **camera**.
        - *pose*: The map coordinate system.
    - **label**
      - type: *semantic2d*
      - shape: `256, 512`
    - **depth**
      - type: *depth*
      - shape: `256, 512`
      - normalize: `true`
      - range: `0, 80`
      - frame-id: The camera coordinate system.
    - **pose_err**

      If not set, it will be generated automatically during validation and evaluation.
      - type: *pose*
      - frame-id: The camera coordinate system that includes the self-localization error.
      - from:
        - *pose*: The camera coordinate system that **does not** includes the self-localization error.
