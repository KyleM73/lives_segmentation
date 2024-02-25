# lives_segmentation
LiDAR InformEd Visual Search (LIVES) with Learned Scan Classification

## Setup
- clone this repo and `cd` into it
- download and unzip the `csv` folder [here](https://drive.google.com/drive/folders/1-eR46L3ezZ0tqjVDVeNlBWyfL8e5p_no?usp=sharing) and `cp` it into `lives_segmentation/data`
- `conda create -n lives python=3.10 && conda activate lives`
- `pip install -r requirements`
- `python train.py`
    - to train with `cuda` or `mps`, edit the device in `settings.yaml`
