# Project 1 Image Enhancement: Single Image Haze Removal

## (Bronte) Sihan Li, Cole Crescas, Karan Shah

## Environment

Training tasks are run in a Linux environment with a CUDA GPU.

## Requirements

To install dependencies, run:

    pip install pytorch==1.10.2 torchvision torchaudio cudatoolkit==11.3
    pip install -r requirements.txt
    pip install image_dehazer

### Dehaze.ipynb
- The main file for the project. It contains the code for data processing as well as training and testing the dehazeformer model.

### train_test_dehazeformer.py
- Wrapper script for training and testing the different dehazeformer models and write results to disc.
- Usage:
  - For training and testing:
  
        python train_test_dehazeformer.py --model dehazeformer-s

  - For test only on a pretrained model:

        python train_test_dehazeformer.py --model dehazeformer-s --test_only

### simple_dehaze.py
- Simple implementation of dehazing algorithm using boundary constraint on the transmission function.

