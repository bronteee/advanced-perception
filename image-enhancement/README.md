# Project 1 Image Enhancement: Single Image Haze Removal

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

