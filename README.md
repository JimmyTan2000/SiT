# This repository is modified based on the original [SiT repository](https://github.com/willisma/SiT). 
This is the repository of the Image and Video Synthesis SoSe 2025 Practical for Pose Estimation + Novel View Synthesis. 

## What has been modified? 
The oiginal SiT repository is about the training and inference on the model based on the ImageNet 256 x 256 dataset that contains 1000 classes. We have adapted it to perform the following:

- [x] Adapt the training script (train.py) for single class.  
- [x] Adapt the training script so that it ingest directly the tensors for training instead of jpg files (using dataset.py). 
- [x] Shift the gaussian mean to poses instead of mean 0. 
- [x] Implement bf16 training to save CUDA memory. 
- [x] Separate the evaluation from the training loop to prevent CUDA out of memory. 
- [x] Implement evaluation of the trained model (evaluate.ipynb). 
- [x] Create turntable dataset for model evaluation (refer to [this repository](https://github.com/willisma/SiT)). 
- [x] Implement notebook for interpolation experiment (interpolation_experiment.ipynb) with the trained model. 
- [x] Implement flow reversal. See "Test Invertibility of SiT" part in `evaluate.ipynb` and "Pose Recovery and Error Evaluation" part in `scene_prediction_loop.ipynb`.

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

Run the following command (for a single GPU setup)
```bash 
torchrun --nnodes=1 --nproc_per_node=1 train.py --model SiT-B/8 --num-classes 1 --epochs 300 --global-batch-size 150 --num-workers 0 --ckpt-every 4000 --cfg-scale 1.0 --image-size 128
```
Note: The dataset is contained in the `output_images` directory and it is generated using [this repository](https://github.com/JimmyTan2000/nerf-image-generation).

**For more details, please refer to the original [SiT repository](https://github.com/willisma/SiT) as this is still incomplete.** 