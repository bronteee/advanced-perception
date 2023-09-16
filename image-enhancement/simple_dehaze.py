# Bronte Sihan Li, 2023
# This is largely based on the work of: https://github.com/Utkarsh-Deshmukh/Single-Image-Dehazing-Python/

import image_dehazer
import cv2
import os

haze_dir = 'data/a2i2/UAV-train/paired_dehaze/images/hazy/'
result_dir = 'results/image_dehazer/'
os.makedirs(result_dir, exist_ok=True)
images_to_sample = [
    '060.png',
    '074.png',
    '080.png',
]


for image in images_to_sample:
    hazy_img = cv2.imread(haze_dir + image)
    hazy_corrected, haze_map = image_dehazer.remove_haze(hazy_img)
    cv2.imshow('input', hazy_img)
    cv2.imshow('hazy_corrected', hazy_corrected)
    cv2.imshow('haze_map', haze_map)
    # cv2.waitKey(0)
    cv2.imwrite(result_dir + image, hazy_corrected)

### user controllable parameters (with their default values):
airlightEstimation_windowSze = 15
boundaryConstraint_windowSze = 3
C0 = 20
C1 = 300
regularize_lambda = 0.1
sigma = 0.5
delta = 0.85
showHazeTransmissionMap = True
