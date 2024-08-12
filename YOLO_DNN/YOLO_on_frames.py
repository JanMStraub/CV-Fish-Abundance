# -*- coding: utf-8 -*-
"""
YOLO on Frames Processing Script
================================

This script processes video frames and their corresponding ground truth annotations to generate mixed images using GMM and Optical Flow results.
It saves the processed images for further analysis.

Author: ahsanjalal, Jan M. Straub
Date: 2024-08-12

Constants:
- GT_DIR: Directory containing the ground truth annotated frames.
- GMM_RESULTS: Directory containing the GMM output images.
- OPTICAL_RESULTS: Directory containing the Optical Flow output images.
- SAVE_MAIN_DIR: Directory to save the mixed images.
- SPECIE_LIST: List of fish species to be labeled.

Functions:
- process_video: Processes a single video's frames and their corresponding ground truth annotations.
- process_gt_file: Processes a single ground truth file and generates a mixed image.
- save_image: Saves the processed mixed image to the specified directory.
- main: Main function to process all video folders in the GT_DIR directory.

Usage:
- Run this script to process all video folders in the GT_DIR and save the results in SAVE_MAIN_DIR.
"""

import os
import glob
import numpy as np
import cv2

# Constants
GT_DIR = "~/annotated_frames"
GMM_RESULTS = "/Users/jan/Documents/code/cv/project/train_gmm"
OPTICAL_RESULTS = "~/Optical_flow"
SAVE_MAIN_DIR = "~/no_gray_gmm_optical_mixed"
SPECIE_LIST = [
    "abudefduf vaigiensis",
    "acanthurus nigrofuscus",
    "amphiprion clarkii",
    "chaetodon lununatus",
    "chaetodon speculum",
    "chaetodon trifascialis",
    "chromis chrysura",
    "dascyllus aruanus",
    "dascyllus reticulatus",
    "hemigumnus malapterus",
    "myripristis kuntee",
    "neoglyphidodon nigroris",
    "pempheris vanicolensis",
    "plectrogly-phidodon dickii",
    "zebrasoma scopas",
    "Background",
]

# Global variables
bkg_count = 0
total_gt_count = 0
TP = 0
FP = 0
gmm_count = 0
num = np.zeros(16)  # 17 for UWA dataset
vid_counter = 0


def process_video(video_fol):
    """
    Processes a single video's frames and their corresponding ground truth annotations.

    Parameters:
    - video_fol: Folder name of the video.

    This function reads the ground truth text files and corresponding images, and processes each frame.
    """
    global total_gt_count
    vid_fol_path = os.path.join(GT_DIR, video_fol)
    os.chdir(vid_fol_path)
    video_name = video_fol.split(".flv")[0]
    gt_text_files = glob.glob("*.txt")
    gt_height, gt_width = [640, 640]
    gmm_height, gmm_width = [640, 640]

    for gt_file in gt_text_files:
        img_gt = cv2.imread(gt_file.split(".")[0] + ".png")
        with open(gt_file) as f:
            gt_text = f.readlines()
        gt_count = len(gt_text)
        total_gt_count += gt_count

        process_gt_file(video_fol, gt_file, img_gt)


def process_gt_file(video_fol, gt_file, img_gt):
    """
    Processes a single ground truth file and generates a mixed image.

    Parameters:
    - video_fol: Folder name of the video.
    - gt_file: Ground truth text file name.
    - img_gt: Ground truth image.

    This function reads the GMM and Optical Flow images, combines them with the ground truth image, and saves the result.
    """
    gmm_img_path = (
        os.path.join(GMM_RESULTS, video_fol, gt_file).split(".txt")[0] + ".png"
    )
    optical_img_path = (
        os.path.join(OPTICAL_RESULTS, video_fol, gt_file).split(".txt")[0] + ".png"
    )

    if os.path.isfile(gmm_img_path):
        img_gmm = cv2.imread(gmm_img_path)
        img_optical = cv2.imread(optical_img_path)
        img_optical = cv2.resize(img_optical, [640, 640])
        img_gt_gray = cv2.cvtColor(img_gt, cv2.COLOR_BGR2GRAY)
        img_gt[:, :, 0] = 0
        img_gt[:, :, 1] = img_gmm[:, :, 0]
        img_gt[:, :, 2] = img_optical[:, :, 0]
    else:
        img_gmm = np.zeros((640, 640))
        if os.path.isfile(optical_img_path):
            img_optical = cv2.imread(optical_img_path)
            img_optical = cv2.resize(img_optical, [640, 640])
        else:
            img_optical = np.zeros((640, 640, 3))

        img_gt_gray = cv2.cvtColor(img_gt, cv2.COLOR_BGR2GRAY)
        img_gt[:, :, 0] = 0
        img_gt[:, :, 1] = img_gmm
        img_gt[:, :, 2] = img_optical[:, :, 0]

    save_image(video_fol, gt_file, img_gt)


def save_image(video_fol, gt_file, img_gt):
    """
    Saves the processed mixed image to the specified directory.

    Parameters:
    - video_fol: Folder name of the video.
    - gt_file: Ground truth text file name.
    - img_gt: Processed mixed image.

    This function creates the save directory if it doesn't exist and saves the image.
    """
    save_path = os.path.join(SAVE_MAIN_DIR, video_fol)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    cv2.imwrite(os.path.join(save_path, gt_file).split(".txt")[0] + ".png", img_gt)


def main():
    """
    Main function to process all video folders in the GT_DIR directory.

    This function performs the following steps:
    1. Retrieves a list of video folders in the GT_DIR directory.
    2. Processes each video folder and its corresponding ground truth annotations.
    """
    global vid_counter
    gt_fol = os.listdir(GT_DIR)
    for video_fol in gt_fol:
        print(f"video number {vid_counter} is in process and video is {video_fol}")
        vid_counter += 1
        process_video(video_fol)


if __name__ == "__main__":
    """
    Entry point of the script.

    This block checks if the script is being run directly (not imported as a module).
    If so, it calls the main() function to start processing the video folders.
    """
    main()
