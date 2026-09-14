# Low-rank Continual Learning

[![ICLR 2026](https://img.shields.io/badge/ICLR-2026-purple.svg)](https://iclr.cc/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Implementation-blue.svg)](https://pytorch.org/)

📄 **Paper:** [Revisiting Weight Regularization for Low-Rank Continual Learning](https://arxiv.org/abs/2602.17559)

In this paper, we revisit weight regularization in low-rank CL as a new perspective for mitigating task interference in PECL. Unlike existing low-rank CL methods, we mitigate task interference by regularizing a shared low-rank update through EWC, thereby keeping the storage requirement and inference costs constant regardless of the number of tasks. Moreover, we provide the first systematic investigation of EWC in low-rank CL, showing that it achieves a better stability–plasticity trade-off than other low-rank methods and enables competitive performance across a wide range of trade-off points.

If you find this repository useful for your research, please consider citing:
```
@article{zheng2026revisiting,
  title={Revisiting Weight Regularization for Low-Rank Continual Learning},
  author={Zheng, Yaoyue and Zhang, Yin and van de Weijer, Joost and van de Ven, Gido M and Du, Shaoyi and Zhang, Xuetao and Tian, Zhiqiang},
  journal={arXiv preprint arXiv:2602.17559},
  year={2026}
}
```

![Overview](overview.png)


## Overview

This repository provides a PyTorch implementation of continual learning methods based on Low-rank Adaptation (LoRA). It includes implementations and extensions of the following approaches:

- InfLoRA
- SD-LoRA
- CL-LoRA
- EWC-LoRA
- [New methods will be updated!]


## Requirements

This project is implemented in PyTorch and tested with the following environment:

- python == 3.9.4
- torch == 2.0.1
- torchvision == 0.15.2
- timm == 0.6.12


## Install

You can set up the environment using the provided `environment.yaml` file:

```bash
conda env create -f environment.yaml
```

Or install manually:

```bash
conda create -n low-rank-cl python=3.9.4 pytorch=2.0.1 torchvision=0.15.2 pytorch-cuda=11.8 -c pytorch -c nvidia
conda activate low-rank-cl
pip install six ipdb scipy pyyaml tqdm timm==0.6.12 numpy==1.25.2 scikit-learn==1.2.0
```


## Dataset

- CIFAR-100: will be downloaded automatically

- DomainNet: download dataset from https://ai.bu.edu/M3SDA/

- ImageNet-R: download dataset from https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar

- ImageNet-A: download dataset from https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar

After downloading, place the datasets in your preferred location and create a symbolic link named `data` in the project root directory:

```bash
ln -s /your/dataset/path ./data
```

Alternatively, you can modify the dataset paths directly in the configuration files.


## Training

All experiments were tested on 1-2 Quadro RTX 6000 GPUs (24GB each).

### Quick Start

Run a quick experiment on CIFAR-100 with:

```bash
python main.py --device [GPU_ID] --config configs/[method]/cifar100.json
```

Example:

```bash
python main.py --device 0 --config configs/ewclora/cifar100.json
python main.py --device 5 --config configs/sdlora_moe/cifar100_seed=42.json
python main.py --device 5 --config configs/sdlora_moe/cifar100_seed=42.json
python main.py --device 0 --config configs/splitlora/cifar100_seed=1993.json
python main.py --device 0 --config configs/splitlora_improvedv4/cifar100_seed=1993.json
python main.py --device 0 --config configs/lorasub_drs/cifar100.json
python main.py --device 0  --config configs/splitlorav6/imagenet-a_seed=1993.json



```

By default, only log files are saved. Model checkpoints are NOT saved to reduce disk usage. To enable checkpoint saving, use the `--save_ckp` flag:

```bash
python main.py --device 0 --config configs/ewclora/cifar100.json --save_ckp
```

For ImageNet-R and ImageNet-A, the train/test split will be processed automatically.


## Acknowledgments

This implementation is inspired by and builds upon the following excellent projects: 

- [PILOT](https://github.com/LAMDA-CL/LAMDA-PILOT)
- [InfLoRA](https://github.com/liangyanshuo/InfLoRA)
- [SD-LoRA](https://github.com/WuYichen-97/SD-Lora-CL)
- [CL-LoRA](https://github.com/JiangpengHe/CL-LoRA)

Many thanks for these great works!
