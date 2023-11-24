
# PanBench: Towards High-Resolution and High-Performance Pansharpening

<p align="left">
<a href="https://arxiv.org/abs/2306.11249" alt="arXiv">
    <img src="https://img.shields.io/badge/arXiv-2311.12083-b31b1b.svg?style=flat" /></a>
<a href="https://github.com/chengtan9907/OpenSTL/blob/master/LICENSE" alt="license">
    <img src="https://img.shields.io/badge/license-Apache--2.0-%23002FA7" /></a>
<!-- <a href="https://huggingface.co/OpenSTL" alt="Huggingface">
    <img src="https://img.shields.io/badge/huggingface-OpenSTL-blueviolet" /></a> -->
<a href="https://openstl.readthedocs.io/en/latest/" alt="docs">
    <img src="https://readthedocs.org/projects/openstl/badge/?version=latest" /></a>
<a href="https://github.com/chengtan9907/OpenSTL/issues" alt="docs">
    <img src="https://img.shields.io/github/issues-raw/chengtan9907/SimVPv2?color=%23FF9600" /></a>
<a href="https://github.com/chengtan9907/OpenSTL/issues" alt="resolution">
    <img src="https://img.shields.io/badge/issue%20resolution-1%20d-%23B7A800" /></a>
<a href="https://img.shields.io/github/stars/chengtan9907/OpenSTL" alt="arXiv">
    <img src="https://img.shields.io/github/stars/chengtan9907/OpenSTL" /></a>
</p>

[📘Documentation](https://openstl.readthedocs.io/en/latest/) |
[🛠️Installation](docs/en/install.md) |
[🚀Model Zoo](docs/en/model_zoos/video_benchmarks.md) |
[🤗Huggingface](https://huggingface.co/OpenSTL) |
[👀Visualization](docs/en/visualization/video_visualization.md) |
[🆕News](docs/en/changelog.md)

## Introduction

Deep learning techniques have shown significant success in pansharpening, existing methods often face limitations in their evaluation, focusing on restricted satellite data sources, single scene types, and low-resolution images. This paper addresses this gap by introducing PanBench, a high-resolution multi-scene dataset containing all mainstream satellites and comprising 5,898 pairs of samples. Each pair includes a four-channel (RGB + near-infrared) multispectral image of 256$\times$256 pixels and a mono-channel panchromatic image of 1,024X1,024 pixels. To achieve high-fidelity synthesis, we propose a Cascaded Multiscale Fusion Network (CMFNet) for Pansharpening. Extensive experiments validate the effectiveness of CMFNet. We have released the dataset, source code, and pre-trained models in the supplementary, fostering further research in remote sensing.

<p align="center" width="100%">
  <img src='https://github-production-user-asset-6210df.s3.amazonaws.com/44519745/246222226-61e6b8e8-959c-4bb3-a1cd-c994b423de3f.png' width="90%">
</p>

<p align="right">(<a href="#top">back to top</a>)</p>

## Overview

<details open>
<summary>Major Features and Plans</summary>

- **Flexable Code Design.**
  OpenSTL decomposes STL algorithms into `methods` (training and prediction), `models` (network architectures), and `modules`, while providing unified experiment API. Users can develop their own STL algorithms with flexible training strategies and networks for different STL tasks.

- **Standard Benchmarks.**
  OpenSTL will support standard benchmarks of STL algorithms image with training and evaluation as many open-source projects (e.g., [MMDetection](https://github.com/open-mmlab/mmdetection) and [USB](https://github.com/microsoft/Semi-supervised-learning)). We are working on training benchmarks and will update results synchronizingly.

- **Plans.**
  We plan to provide benchmarks of various STL methods and MetaFormer architectures based on SimVP in various STL application tasks, e.g., video prediction, weather prediction, traffic prediction, etc. We encourage researchers interested in STL to contribute to OpenSTL or provide valuable advice!

</details>

<details open>
<summary>Code Structures</summary>

- `openstl/api` contains an experiment runner.
- `openstl/core` contains core training plugins and metrics.
- `openstl/datasets` contains datasets and dataloaders.
- `openstl/methods/` contains training methods for various video prediction methods.
- `openstl/models/` contains the main network architectures of various video prediction methods.
- `openstl/modules/` contains network modules and layers.
- `tools/` contains the executable python files `tools/train.py` and `tools/test.py` with possible arguments for training, validating, and testing pipelines.

</details>

## News and Updates

[2023-09-23] PanBench: Towards High-Resolution and High-Performance Pansharpening! [arXiv](https://arxiv.org/abs/2311.12083).

[2023-06-19] `OpenSTL` v0.3.0 is released and will be enhanced in [#25](https://github.com/chengtan9907/OpenSTL/issues/25).

## Installation

This project has provided an environment setting file of conda, users can easily reproduce the environment by the following commands:
```shell
git clone https://github.com/chengtan9907/OpenSTL
cd OpenSTL
conda env create -f environment.yml
conda activate OpenSTL
python setup.py develop
```

<details close>
<summary>Dependencies</summary>

* argparse
* dask
* decord
* fvcore
* hickle
* lpips
* matplotlib
* netcdf4
* numpy
* opencv-python
* packaging
* pandas
* python<=3.10.8
* scikit-image
* scikit-learn
* torch
* timm
* tqdm
* xarray==0.19.0
</details>

Please refer to [install.md](docs/en/install.md) for more detailed instructions.

## Getting Started

Please see [get_started.md](docs/en/get_started.md) for the basic usage. Here is an example of single GPU non-distributed training SimVP+gSTA on Moving MNIST dataset.
```shell
bash tools/prepare_data/download_mmnist.sh
python tools/train.py -d mmnist --lr 1e-3 -c configs/mmnist/simvp/SimVP_gSTA.py --ex_name mmnist_simvp_gsta
```

## Tutorial on using Custom Data

For the convenience of users, we provide a tutorial on how to train, evaluate, and visualize with OpenSTL on custom data. This tutorial enables users to quickly build their own projects using OpenSTL. For more details, please refer to the [`tutorial.ipynb`](examples/tutorial.ipynb) in the `examples/` directory.

We also provide a Colab demo of this tutorial:

<a href="https://colab.research.google.com/drive/19uShc-1uCcySrjrRP3peXf2RUNVzCjHh?usp=sharing" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

<p align="right">(<a href="#top">back to top</a>)</p>

## Overview of Model Zoo and Datasets

We support various spatiotemporal prediction methods and provide [benchmarks](https://github.com/chengtan9907/OpenSTL/tree/master/docs/en/model_zoos) on various STL datasets. We are working on add new methods and collecting experiment results.

* Spatiotemporal Prediction Methods.

    <details open>
    <summary>Pansharpening Methods</summary>

    - [x] [PNN](https://www.mdpi.com/2072-4292/8/7/594) (Remote Sensing'2016)
    - [x] [PanNet](https://xueyangfu.github.io/paper/2017/iccv/YangFuetal2017.pdf) (ICCV'2017)
    - [x] [MSDCNN](https://arxiv.org/abs/1712.09809) (J-STARS'2018)
    - [x] [TFNet](https://arxiv.org/abs/1711.02549) (Inform. Fusion'2020)
    - [x] [FusionNet](https://ieeexplore.ieee.org/abstract/document/9240949) (TGRS'2020)
    - [x] [GPPNN](https://arxiv.org/abs/2103.04584) (CVPR'2021)
    - [x] [SRPPNN](https://ieeexplore.ieee.org/abstract/document/9172104) (TGRS'2021)
    - [x] [PGCU](https://arxiv.org/abs/2303.13659) (CVPR'2023)

    
    </details>

* Spatiotemporal Predictive Learning Benchmarks ([prepare_data](https://github.com/chengtan9907/OpenSTL/tree/master/tools/prepare_data) or [Baidu Cloud](https://pan.baidu.com/s/1fudsBHyrf3nbt-7d42YWWg?pwd=kjfk)).

    <details open>
    <summary>Currently supported satellite</summary>

    - [x] [GaoFen1]
    - [x] [GaoFen2]
    - [x] [GaoFen6]
    - [x] [Landsat7]
    - [x] [Landsat8]
    - [x] [WorldView2]
    - [x] [WorldView3]
    - [x] [WorldView4]
    - [x] [QuickBird]
    - [x] [IKONOS]

    </details>

<p align="right">(<a href="#top">back to top</a>)</p>

## Visualization

We present visualization examples of ConvLSTM below. For more detailed information, please refer to the [visualization](docs/en/visualization/).

- For synthetic moving object trajectory prediction and real-world video prediction, visualization examples of other approaches can be found in [visualization/video_visualization.md](docs/en/visualization/video_visualization.md). BAIR and Kinetics are not benchmarked and only for illustration.

- For traffic flow prediction, visualization examples of other approaches are shown in [visualization/traffic_visualization.md](docs/en/visualization/traffic_visualization.md).

- For weather forecasting, visualization examples of other approaches are shown in [visualization/weather_visualization.md](docs/en/visualization/weather_visualization.md).

<div align="center">

| Moving MNIST | Moving FMNIST | 
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/mmnist_ConvLSTM.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/moving_fashionmnist_ConvLSTM.gif' height="auto" width="260" ></div> |

| Moving MNIST-CIFAR | KittiCaltech |
| :---: | :---: |
|  <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/moving_mnist_cifar_ConvLSTM.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/kitticaltech_ConvLSTM.gif' height="auto" width="260" ></div> |

| KTH | Human 3.6M | 
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/kth20_ConvLSTM.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/human_ConvLSTM.gif' height="auto" width="260" ></div> |

| Traffic - in flow | Traffic - out flow |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-traffic/taxibj_in_flow_ConvLSTM.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-traffic/taxibj_out_flow_ConvLSTM.gif' height="auto" width="260" ></div> |

| Weather - Temperature | Weather - Humidity |
|  :---: |  :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_temperature_5_625_ConvLSTM.gif' height="auto" width="360" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_humidity_5_625_ConvLSTM.gif' height="auto" width="360" ></div>|

| Weather - Latitude Wind | Weather - Cloud Cover | 
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_wind_latitude_ConvLSTM.gif' height="auto" width="360" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_cloud_cover_5_625_ConvLSTM.gif' height="auto" width="360" ></div> |

| BAIR Robot Pushing | Kinetics-400 | 
| :---: | :---: |
| <div align=center><img src='https://github-production-user-asset-6210df.s3.amazonaws.com/44519745/257872182-4f31928d-2ebc-4407-b2d4-1fe4a8da5837.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github-production-user-asset-6210df.s3.amazonaws.com/44519745/257872560-00775edf-5773-478c-8836-f7aec461e209.gif' height="auto" width="260" ></div> |

</div>

## License

This project is released under the [Apache 2.0 license](LICENSE). See `LICENSE` for more information.
