# Perceptual Losses for Real-Time Style Transfer and Super-Resolution

This repository contains a pytorch implementation of an algorithm for artistic style transfer. The algorithm aims mixing the content of an image with the style of another image. The model uses the method decribed in [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155) along with [Instance Normalization](https://arxiv.org/pdf/1607.08022.pdf). I already included the trained models for a quick demonstration.

## Results
![cat](./results/220206_Style_Transfer.png)

## Requirements
Setting up for this project involves installing dependencies and preparing datasets. The code is tested on Ubuntu 20.04 with NVIDIA GPUs and CUDA installed. 

To install all dependencies, please run the following:
```bash
# CUDA 11.0
python3 -m pip install pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

# CPU only
python3 -m pip install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

# other dependencies
python3 -m pip install -r requirements.txt
```

## Quick Demonstration
I prepared [a jupyter notebook](https://github.com/OFRIN/Fast_Neural_Style_Transfer/demo.ipynb).

## Usage
Train a stylizing model.
```bash
python3 train.py \
--gpus 0 --root_dir ../first_ssd_8tb/COCO2014/train/image/ \
--style_path ./data/ben_giles.jpg \
--tag Transformer@ben_giles --max_epochs 10
```
* `--root_dir` : path to a directory including all the training images. I used COCO 2014 training images [80K/13GB] [(download)](https://cocodataset.org/#download).
* `--style_path` : path to a style image.
* `--tag` : name to be used for saving a weight file.

Stylize an image through the trained weight.
```bash
python3 test.py \
--image_path ./samples/SSBO_1.jpg \
--tag Transformer@ben_giles
```
* `--image_path` : path to a content image you want to stylize.
* `--tag` : name to be used for stylizing an image.

Convert pth file to pt file for an inference without a graphic card.
```bash
python3 convert_pth_to_pt.py \
--pth_path ./experiments/models/Transformer@vc_monariza/ep=2.pth \
--pt_path ./weights/Transformer@vc_monariza.pt
```
