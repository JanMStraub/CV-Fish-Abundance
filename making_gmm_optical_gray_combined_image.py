# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 16:56:16 2018

This script processes video frames by combining GMM and Optical Flow results 
with ground truth images. The combined images are saved in a specified directory.

@author: ahsanjalal
"""

import os
import glob
import numpy as np
import cv2
from pathlib import Path

BASE_DIR = Path("/content/drive/MyDrive/Colab Notebooks/CV_Project")
VIDEO_DIR = BASE_DIR / "data/fishclef_2015_release/training_set/videos"
GT_DIR = BASE_DIR / "data/fishclef_2015_release/training_set/gt"
IMG_DIR = BASE_DIR / "train_img/"
GMM_DIR = BASE_DIR / "train_gmm/"
OPTICAL_DIR = BASE_DIR / "train_optical/"
GMM_OPTICAL_DIR = BASE_DIR / "train_gmm_optical/"
FRAME_RESIZE = (640, 640)  # Dimensions to resize frames

def process_video_gmm_optical(video_path):
    """
    Process a single video folder by reading ground truth and corresponding GMM and Optical Flow images.

    Parameters:
    - video_path: The name of the video folder to process.
    """
    
    vid_fol_path = os.path.join(GT_DIR, video_path)
    os.chdir(vid_fol_path)
    video_name = video_path.split(".flv")[0]
    gt_text_files = glob.glob("*.txt")
    gt_height, gt_width = [640, 640]
    gmm_height, gmm_width = [640, 640]

    for gt_file in gt_text_files:
        img_gt = cv2.imread(gt_file.split(".")[0] + ".png")
        with open(gt_file) as f:
            gt_text = f.readlines()
        gt_count = len(gt_text)

        process_gt_file(video_path, gt_file, img_gt)

def process_gt_file(video_path, gt_file, img_gt):
    """
    Process a single ground truth file by combining it with GMM and Optical Flow images.

    Parameters:
    - video_path: The name of the video folder to process.
    - gt_file: The name of the ground truth file.
    - img_gt: The ground truth image.
    """
    gmm_img_path = (
        os.path.join(GMM_DIR, video_path, gt_file).split(".txt")[0] + ".png"
    )
    optical_img_path = (
        os.path.join(OPTICAL_DIR, video_path, gt_file).split(".txt")[0] + ".png"
    )

    if os.path.isfile(gmm_img_path):
        img_gmm = cv2.imread(GMM_DIR)
        img_gmm = cv2.resize(img_gmm, FRAME_RESIZE)
        
        img_optical = cv2.imread(OPTICAL_DIR)
        img_optical = cv2.resize(img_optical, FRAME_RESIZE)
        
        img_gt_gray = cv2.cvtColor(IMG_DIR, cv2.COLOR_BGR2GRAY)
        img_gt_gray = cv2.resize(img_gt_gray, FRAME_RESIZE)
        
        img_gt[:, :, 0] = img_gt_gray[:, :, 0]
        img_gt[:, :, 1] = img_gmm[:, :, 0]
        img_gt[:, :, 2] = img_optical[:, :, 0]
    else:
        img_gmm = np.zeros((640, 640))
        if os.path.isfile(optical_img_path):
            img_optical = cv2.imread(optical_img_path)
            img_optical = cv2.resize(img_optical, FRAME_RESIZE)
            
            img_gt_gray = cv2.cvtColor(IMG_DIR, cv2.COLOR_BGR2GRAY)
            img_gt_gray = cv2.resize(img_gt_gray, FRAME_RESIZE)
        else:
            img_optical = np.zeros((640, 640, 3))

        img_gt[:, :, 0] = img_gt_gray[:, :, 0]
        img_gt[:, :, 1] = img_gmm[:, :, 0]
        img_gt[:, :, 2] = img_optical[:, :, 0]

    save_image(video_path, gt_file, img_gt)

def save_image(video_path, gt_file, img_gt):
    """
    Save the combined image to the specified directory.

    Parameters:
    - video_path: The name of the video folder to process.
    - gt_file: The name of the ground truth file.
    - img_gt The combined image to save.
    """
    save_path = os.path.join(GMM_OPTICAL_DIR, video_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    cv2.imwrite(os.path.join(save_path, gt_file).split(".txt")[0] + ".png", img_gt)

def main():
    """
    Function to process all video folders in the ground truth directory.
    """
    
    video_files = [
        f for f in os.listdir(VIDEO_DIR) if f.endswith(".flv") or f.endswith(".avi")
    ]
    
    for idx, video_file in enumerate(video_files):
        print(f"Processing video {idx + 1}/{len(video_files)}: {video_file.name}")
        process_video_gmm_optical(video_file)


if __name__ == "__main__":
    main()