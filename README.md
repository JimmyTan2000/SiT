# This repository is modified based on the original [SiT repository](https://github.com/willisma/SiT). 
This is the repository of the Image and Video Synthesis SoSe 2025 Practical for Pose Estimation + Novel View Synthesis. 

## What has been modified? 
The oiginal SiT repository is about the training and inference on the model based on the ImageNet 256 x 256 dataset that contains 1000 classes. We have adapted it to train and generate images of a single class. This is not the final version yet so there is a TODO list to show what has been done and what hasn't. 

### TODO List 
- [x] Write a script to convert generated tensors into JPG (experimental only). [Refer to this notebook.](https://github.com/JimmyTan2000/nerf-image-generation)
- [x] Adapt the training script (train.py) for single class. 
- [x] Adapt the inference notebook (run_SiT.ipynb) to the trained model. 
- [ ] Adapt the training script so that it ingest directly the tensors for training instead of jpg files. 
- [ ] Shift the gaussian mean to poses instead of mean 0. 
- [ ] Write a shell script for training using SLURM to make use of the CIP Pool. 

## Setup
Step 1: Download and set up the repo:
```bash 
git clone https://github.com/JimmyTan2000/SiT.git
cd SiT
```
Step 2: Create a conda environment:
```bash
conda env create -f environment.yml
conda activate SiT
```
If you only want to run pre-trained models locally on CPU, you can remove the `cudatoolkit` and `pytorch-cuda` requirements from the environment.yml file.

## Training SiT
Step 1: Set the environment variables:
```bash
export WANDB_KEY="key"
export ENTITY="entity name"
export PROJECT="project name"
```

Step 2: Run the following command (for a single GPU setup)
```bash 
torchrun --nnodes=1 --nproc_per_node=1 train.py --model SiT-S/2 --data-path output_images_128 --num-classes 1 --epochs 200 --global-batch-size 50 --num-workers 1 --ckpt-every 2000 --cfg-scale 1.0 --image-size 128 --ckpt-every 2000 --cfg-scale 1.0
```
Note: The dataset is contained in the `output_images` directory and it is generated using [this repository](https://github.com/JimmyTan2000/nerf-image-generation).

### Additional commands: 
- `Interpolant settings (Haven't tried before)`:
We also support different choices of interpolant and model predictions. For example, to launch SiT-XL/2 (256x256) with Linear interpolant and noise prediction:
```bash
torchrun --nnodes=1 --nproc_per_node=1 train.py --model SiT-XL/2 --data-path /path/to/imagenet/train --path-type Linear --prediction noise
```
- `Resume training:`To resume training from custom checkpoint:
```bash
torchrun --nnodes=1 --nproc_per_node=1 train.py --model SiT-L/2 --data-path /path/to/imagenet/train --ckpt /path/to/model.pt
```

**For more details, please refer to the original [SiT repository](https://github.com/willisma/SiT) as this is still incomplete.** 