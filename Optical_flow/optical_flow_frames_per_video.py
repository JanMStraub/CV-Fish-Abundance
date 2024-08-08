# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 11:48:07 2018

Author: ahsanjalal
"""

import os
import cv2
import numpy as np

# Constants
VIDEO_DIR = "~/Training_dataset/Videos/"
SAVE_DIR = "~/Train_Optical_flow/"
KERNEL_SIZE = (7, 7)
FOURCC = cv2.VideoWriter_fourcc(*"MJPG")


def setup_directories(video_file):
    """Create directories for saving optical flow frames if they don't exist."""
    save_path = os.path.join(SAVE_DIR, video_file)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    return save_path


def preprocess_frame(frame):
    """Convert frame to YUV, equalize histogram, and convert back to BGR."""
    img_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)


def calculate_optical_flow(prvs, next1):
    """Calculate optical flow using Farneback method."""
    return cv2.calcOpticalFlowFarneback(prvs, next1, None, 0.95, 10, 15, 3, 5, 1.2, 0)


def process_video(video_file):
    """Process a single video to extract and save optical flow frames."""
    video_name = os.path.splitext(video_file)[0]
    save_path = setup_directories(video_name)
    cap = cv2.VideoCapture(os.path.join(VIDEO_DIR, video_file))
    ret, frame1 = cap.read()
    if not ret:
        print(f"Failed to read the first frame of {video_file}")
        return

    frame1 = preprocess_frame(frame1)
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255
    kernel = np.ones(KERNEL_SIZE, np.uint8)
    frame_count = 0

    while True:
        ret, frame2 = cap.read()
        if not ret:
            break

        frame2 = preprocess_frame(frame2)
        next1 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        frame_count += 1
        print(frame_count)

        flow = calculate_optical_flow(prvs, next1)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        opening = cv2.morphologyEx(rgb, cv2.MORPH_OPEN, kernel)
        opening_gray = cv2.cvtColor(opening, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join(save_path, f"{frame_count:03d}.png"), opening_gray)

        prvs = next1

    cap.release()


def main():
    """Main function to process all videos in the directory."""
    os.chdir(VIDEO_DIR)
    video_names = os.listdir(VIDEO_DIR)
    for video_file in video_names:
        process_video(video_file)


if __name__ == "__main__":
    main()
