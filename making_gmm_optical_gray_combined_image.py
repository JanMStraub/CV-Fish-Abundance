# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 16:56:16 2018

@author: ahsanjalal
"""

import os
import glob
import numpy as np
import cv2

# Constants
GT_DIR = "~/annotated_frames"
GMM_RESULTS = "~/gmm_output"
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
    save_path = os.path.join(SAVE_MAIN_DIR, video_fol)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    cv2.imwrite(os.path.join(save_path, gt_file).split(".txt")[0] + ".png", img_gt)


def main():
    global vid_counter
    gt_fol = os.listdir(GT_DIR)
    for video_fol in gt_fol:
        print(f"video number {vid_counter} is in process and video is {video_fol}")
        vid_counter += 1
        process_video(video_fol)


if __name__ == "__main__":
    main()
