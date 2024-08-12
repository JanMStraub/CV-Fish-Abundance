"""
GMM Frames Per Video Processing Script
======================================

This script processes video files to extract frames and perform Gaussian Mixture Model (GMM) based foreground detection.
It saves the processed frames and annotations for further analysis.

Author: ahsanjalal, Jan M. Straub
Date: 2024-08-12

Constants:
- MAIN_DIR: Directory to save processed frames and annotations.
- VIDEO_DIR: Directory containing the input video files.
- FOREGROUND_DETECTOR_PARAMS: Parameters for the foreground detector.
- BLOB_ANALYSIS_PARAMS: Parameters for blob analysis.
- STRUCTURING_ELEMENT_OPEN: Structuring element for morphological opening.
- STRUCTURING_ELEMENT_CLOSE: Structuring element for morphological closing.
- FRAME_RESIZE: Dimensions to resize frames.
- FRAME_ADJUST_GAMMA: Gamma adjustment value for frames.

Functions:
- adjust_gamma: Adjusts the gamma of an image.
- save_frame_and_annotations: Saves processed frames and annotations.

Usage:
- Run this script to process all videos in the VIDEO_DIR and save the results in MAIN_DIR.
"""

import os
import cv2
import numpy as np

# Constants
MAIN_DIR = "/Users/jan/Documents/code/cv/project/train_gmm"  # Directory to save processed frames and annotations
VIDEO_DIR = "/Users/jan/Documents/code/cv/project/data/fishclef_2015_release/training_set/videos"  # Directory containing the input video files
FOREGROUND_DETECTOR_PARAMS = {
    "history": 250,
    "varThreshold": 16,
    "detectShadows": True,
}  # Parameters for the foreground detector
BLOB_ANALYSIS_PARAMS = {"min_area": 200}  # Parameters for blob analysis
STRUCTURING_ELEMENT_OPEN = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (3, 3)
)  # Structuring element for morphological opening
STRUCTURING_ELEMENT_CLOSE = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (5, 5)
)  # Structuring element for morphological closing
FRAME_RESIZE = (640, 640)  # Dimensions to resize frames
FRAME_ADJUST_GAMMA = 1.5  # Gamma adjustment value for frames


# Function to adjust gamma
def adjust_gamma(image, gamma=1.0):
    """
    Adjusts the gamma of an image.

    Parameters:
    - image: Input image.
    - gamma: Gamma value for adjustment.

    Returns:
    - Adjusted image.
    """
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in np.arange(0, 256)]).astype(
        "uint8"
    )
    return cv2.LUT(image, table)


def save_frame_and_annotations(opfolder, frame_idx, frame, filteredForeground, bbox):
    """
    Saves processed frames and annotations.

    Parameters:
    - opfolder: Output folder path.
    - frame_idx: Index of the current frame.
    - frame: Original frame.
    - filteredForeground: Foreground mask.
    - bbox: Bounding boxes of detected objects.
    """
    opBaseFileName = f"{frame_idx:03d}.png"  # Output file name for the frame
    textfilename = f"{frame_idx:03d}.txt"  # Output file name for the annotations
    opFullFileName = os.path.join(
        opfolder, opBaseFileName
    )  # Full path for the frame file
    opFullFiletext = os.path.join(
        opfolder, textfilename
    )  # Full path for the annotation file

    test_image = np.zeros_like(
        filteredForeground
    )  # Initialize an empty image for the foreground mask
    img_height, img_width = frame.shape[:2]  # Get the dimensions of the frame

    with open(opFullFiletext, "a") as fileID:
        for x, y, w, h in bbox:
            # Fill the test image with the foreground mask within the bounding box
            test_image[y : y + h, x : x + w] = filteredForeground[y : y + h, x : x + w]
            # Normalize the bounding box coordinates
            x_c = (x + w / 2.0) / img_width
            y_c = (y + h / 2.0) / img_height
            w /= img_width
            h /= img_height
            # Write the normalized bounding box coordinates to the annotation file
            fileID.write(f"0 {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")

    # Save the test image
    cv2.imwrite(opFullFileName, test_image)


def save_empty_annotation(opfolder, frame_idx):
    """
    Creates an empty annotation file for a frame with no detected objects.

    Parameters:
    - opfolder: Output folder path.
    - frame_idx: Index of the current frame.
    """
    textfilename = f"{frame_idx:03d}.txt"  # Output file name for the annotation
    opFullFiletext = os.path.join(
        opfolder, textfilename
    )  # Full path for the annotation file
    open(opFullFiletext, "a").close()  # Create an empty file


def main():
    """
    Main function to process video files for foreground detection and save the results.

    This function performs the following steps:
    1. Retrieves a list of video files from the VIDEO_DIR directory.
    2. For each video file:
        a. Prints the video number and name.
        b. Creates an output folder for the processed frames and annotations.
        c. Initializes the background subtractor for foreground detection.
        d. Reads and processes each frame of the video:
            i. Resizes the frame.
            ii. Adjusts the gamma of the frame.
            iii. Applies the foreground detector to the frame.
            iv. Performs morphological operations to clean up the foreground mask.
            v. Finds contours in the foreground mask to detect blobs.
            vi. Saves the frame and annotations if blobs are detected, otherwise saves an empty annotation.
    3. Releases the video reader after processing all frames.

    Constants:
    - VIDEO_DIR: Directory containing the input video files.
    - MAIN_DIR: Directory to save processed frames and annotations.
    - FOREGROUND_DETECTOR_PARAMS: Parameters for the foreground detector.
    - BLOB_ANALYSIS_PARAMS: Parameters for blob analysis.
    - STRUCTURING_ELEMENT_OPEN: Structuring element for morphological opening.
    - STRUCTURING_ELEMENT_CLOSE: Structuring element for morphological closing.
    - FRAME_RESIZE: Dimensions to resize frames.
    - FRAME_ADJUST_GAMMA: Gamma adjustment value for frames.
    """
    # Get list of video files
    video_name_list = [f for f in os.listdir(VIDEO_DIR) if f.endswith(".flv")]

    for vids, video_name in enumerate(video_name_list):
        print(f"Video number {vids + 1} is in process: {video_name}")

        # Create output folder if it doesn't exist
        opfolder = os.path.join(MAIN_DIR, video_name)
        os.makedirs(opfolder, exist_ok=True)

        # Initialize background subtractor and blob analysis
        foregroundDetector = cv2.createBackgroundSubtractorMOG2(
            **FOREGROUND_DETECTOR_PARAMS
        )

        # Initialize video reader
        videoReader = cv2.VideoCapture(os.path.join(VIDEO_DIR, video_name))

        i = -1
        while True:
            ret, frame = videoReader.read()
            if not ret:
                break
            i += 1

            # Preprocess the frame
            frame = cv2.resize(frame, FRAME_RESIZE)  # Resize the frame
            frame = adjust_gamma(frame, FRAME_ADJUST_GAMMA)  # Adjust gamma

            # Detect foreground
            foreground = foregroundDetector.apply(frame)  # Apply foreground detector
            filteredForeground = cv2.morphologyEx(
                foreground, cv2.MORPH_OPEN, STRUCTURING_ELEMENT_OPEN
            )  # Morphological opening
            filteredForeground = cv2.morphologyEx(
                filteredForeground, cv2.MORPH_CLOSE, STRUCTURING_ELEMENT_CLOSE
            )  # Morphological closing

            # Find contours for blob detection
            contours, _ = cv2.findContours(
                filteredForeground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            bbox = []
            for contour in contours:
                if cv2.contourArea(contour) >= BLOB_ANALYSIS_PARAMS["min_area"]:
                    bbox.append(
                        cv2.boundingRect(contour)
                    )  # Get bounding box for each contour

            if bbox:
                save_frame_and_annotations(
                    opfolder, i, frame, filteredForeground, bbox
                )  # Save frame and annotations
            else:
                save_empty_annotation(
                    opfolder, i
                )  # Save empty annotation if no objects detected

        videoReader.release()  # Release the video reader


if __name__ == "__main__":
    """
    Entry point of the script.

    This block checks if the script is being run directly (not imported as a module).
    If so, it calls the main() function to start processing the video files.
    """
    main()
