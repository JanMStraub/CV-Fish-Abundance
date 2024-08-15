"""
Optical Flow Frames Processing Script
=====================================

This script processes video files to compute optical flow for each frame and save the results.
It uses the Farneback method for optical flow computation.

Author: ahsanjalal, JanMStraub
Date: 2024-08-12

Constants:
- VIDEO_DIR: Directory containing the input video files.
- SAVE_DIR: Directory to save the optical flow results.
- FARNEBACK_PARAMS: Parameters for the Farneback optical flow algorithm.

Functions:
- process_video: Processes a single video file to compute optical flow for each frame.
- save_optical_flow: Saves the computed optical flow image to the specified directory.
- main: Main function to process all video files in the VIDEO_DIR directory.

Usage:
- Run this script to process all videos in the VIDEO_DIR and save the optical flow results in SAVE_DIR.
"""

import os
import cv2
import numpy as np
from pathlib import Path

# Constants
VIDEO_DIR = Path(__file__).parent / "../data/fishclef_2015_release/training_set/videos"
SAVE_DIR = Path(__file__).parent / "../train_optical"
FARNEBACK_PARAMS = {
    "pyr_scale": 0.5,
    "levels": 3,
    "winsize": 15,
    "iterations": 3,
    "poly_n": 5,
    "poly_sigma": 1.2,
    "flags": 0,
}


def process_video(video_path):
    """
    Processes a single video file to compute optical flow for each frame.

    Parameters:
    - video_path: Path to the video file.

    This function reads the video file, computes optical flow for each frame using the Farneback method,
    and saves the results.
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame1 = cap.read()
    if not ret:
        print(f"Failed to read the video file: {video_path}")
        return

    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)  # Convert first frame to grayscale
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255  # Set saturation to maximum

    frame_count = 0
    while True:
        ret, frame2 = cap.read()
        if not ret:
            break

        next = cv2.cvtColor(
            frame2, cv2.COLOR_BGR2GRAY
        )  # Convert next frame to grayscale
        flow = cv2.calcOpticalFlowFarneback(
            prvs, next, None, **FARNEBACK_PARAMS
        )  # Compute optical flow

        # Convert flow to HSV
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        save_optical_flow(video_path, frame_count, bgr)  # Save the optical flow image
        prvs = next
        frame_count += 1

    cap.release()


def save_optical_flow(video_path, frame_count, bgr):
    """
    Saves the computed optical flow image to the specified directory.

    Parameters:
    - video_path: Path to the video file.
    - frame_count: Frame number.
    - bgr: Optical flow image in BGR format.

    This function creates the save directory if it doesn't exist and saves the optical flow image.
    """
    video_name = os.path.basename(video_path).split(".")[0]
    save_path = os.path.join(SAVE_DIR, video_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    cv2.imwrite(os.path.join(save_path, f"frame_{frame_count:04d}.png"), bgr)


def main():
    """
    Main function to process all video files in the VIDEO_DIR directory.

    This function performs the following steps:
    1. Retrieves a list of video files in the VIDEO_DIR directory.
    2. Processes each video file to compute optical flow for each frame.
    """
    video_files = [
        f for f in os.listdir(VIDEO_DIR) if f.endswith(".flv") or f.endswith(".avi")
    ]
    for video_file in video_files:
        video_path = os.path.join(VIDEO_DIR, video_file)
        print(f"Processing video: {video_file}")
        process_video(video_path)


if __name__ == "__main__":
    """
    Entry point of the script.

    This block checks if the script is being run directly (not imported as a module).
    If so, it calls the main() function to start processing the video files.
    """
    main()
