![logo](images/logo_v2.png)

# Pansharpening

## Introduction

This repository is the official PyTorch implementation of our paper [Towards Robust Pansharpening: A Large-Scale High-Resolution Multi-Scene Dataset and Novel Approach](https://www.mdpi.com/2072-4292/16/16/2899).

<p align="center" width="100%">
  <img src='images/toolbox.svg' width="90%">
</p>

## Requirements

```bash
pip install -r requirements.txt
```

## Dataset

- [Google Drive](https://drive.google.com/file/d/1fjwvRrCmExk02c5sxGXMoSvGdGL0FbYR/view?usp=drive_link) | [123 Cloud Drive](https://www.123pan.com/s/RKrRVv-DjBJH.html) | [HuggingFace](https://huggingface.co/datasets/XavierJiezou/pansharpening-datasets/blob/main/PanBench.zip)

```bash
PanBench
├─GF1
│  ├─NIR_256
│  ├─PAN_1024
│  └─RGB_256
├─GF2
│  ├─NIR_256
│  ├─PAN_1024
│  └─RGB_256
├─GF6
│  ├─NIR_256
│  ├─PAN_1024
│  └─RGB_256
├─IN
│  ├─NIR_256
│  ├─PAN_1024
│  └─RGB_256
├─LC7
│  ├─NIR_256
│  ├─PAN_1024
│  └─RGB_256
├─LC8
│  ├─NIR_256
│  ├─PAN_1024
│  └─RGB_256
├─QB
│  ├─NIR_256
│  ├─PAN_1024
│  └─RGB_256
├─WV2
│  ├─NIR_256
│  ├─PAN_1024
│  └─RGB_256
├─WV3
│  ├─NIR_256
│  ├─PAN_1024
│  └─RGB_256
└─WV4
    ├─NIR_256
    ├─PAN_1024
    └─RGB_256
```

## Training

```bash
python src/train.py experiment=cmfnet
```

## Evaluation

```bash
python src/eval.py experiment=cmfnet
```

## Pre-trained Models

You can download pre-trained models in [logs/train/runs](logs/train/runs).

## Overview of Model Zoo and Datasets

- Supported methods.
   
  - [x] [PNN](https://www.mdpi.com/2072-4292/8/7/594) (Remote Sensing'2016)
  - [x] [PanNet](https://xueyangfu.github.io/paper/2017/iccv/YangFuetal2017.pdf) (ICCV'2017)
  - [x] [MSDCNN](https://arxiv.org/abs/1712.09809) (J-STARS'2018)
  - [x] [TFNet](https://arxiv.org/abs/1711.02549) (Inform. Fusion'2020)
  - [x] [FusionNet](https://ieeexplore.ieee.org/abstract/document/9240949) (TGRS'2020)
  - [x] [GPPNN](https://arxiv.org/abs/2103.04584) (CVPR'2021)
  - [x] [SRPPNN](https://ieeexplore.ieee.org/abstract/document/9172104) (TGRS'2021)
  - [x] [PGCU](https://arxiv.org/abs/2303.13659) (CVPR'2023)
  - [x] [CMFNet](www.mdpi.com/2072-4292/16/16/2899) (Remote Sensing'2024)

- Supported satellites.

  - [x] GaoFen1
  - [x] GaoFen2
  - [x] GaoFen6
  - [x] Landsat7
  - [x] Landsat8
  - [x] WorldView2
  - [x] WorldView3
  - [x] WorldView4
  - [x] QuickBird
  - [x] IKONOS

## Visualization

```shell
python visualize.py
```

![map](images/rainbow.png)
![vis](images/vis/15_%5B'GF2'%5D_%5B'vegetation'%5D.png)
![vis](images/vis/23_%5B'QB'%5D_%5B'urban,crops,vegetation'%5D.png)
![vis](images/vis/45_%5B'IN'%5D_%5B'urban,vegetation'%5D.png)
![vis](images/vis/66_%5B'IN'%5D_%5B'urban'%5D.png)
![vis](images/vis/84_%5B'QB'%5D_%5B'urban,crops'%5D.png)
![vis](images/vis/327_%5B'GF1'%5D_%5B'water'%5D.png)
![vis](images/vis/540_%5B'GF2'%5D_%5B'vegetation,urban'%5D.png)

## Citation

If you use our code or models in your research, please cite with:

```latex
@Article{cmfnet,
AUTHOR = {Wang, Shiying and Zou, Xuechao and Li, Kai and Xing, Junliang and Cao, Tengfei and Tao, Pin},
TITLE = {Towards Robust Pansharpening: A Large-Scale High-Resolution Multi-Scene Dataset and Novel Approach},
JOURNAL = {Remote Sensing},
VOLUME = {16},
YEAR = {2024},
NUMBER = {16},
}

```
