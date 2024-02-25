# lives_segmentation
LiDAR Informed Visual Search (LIVES) with Learned Scan Classification

## Setup
- clone this repo and `cd` into it
- download the `csv` folder and `cp` it into `lives_segmentation/data`
- `conda create -n lives python=3.10 && conda activate lives`
- `pip install -r requirements`
- `python train.py`
    - to train with `cuda` or `mps`, edit the device in `settings.yaml`
