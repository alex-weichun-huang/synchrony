# Requirements

## To install conda on your Linux machine:

```sh
cd /tmp
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh
```

## Set up the environment with conda:

```sh
conda create --name synchrony python=3.9
conda activate synchrony
```

## Install torch

Please visit the official PyTorch website and follow the <a href="https://pytorch.org/get-started/locally/">installation guide</a> to install torch, torchvision, and torchaudio that works for your system. The following command worked for my machine:

```sh
conda install pytorch=1.13.0 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia
```

## Install Detectron2 for object detection:

```sh
python -m pip install -e detectron2
```

## Install other dependencies:

```sh
python -m pip install -r requirements.txt
```