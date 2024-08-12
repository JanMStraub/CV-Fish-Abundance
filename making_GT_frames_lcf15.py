"""
Fish Species Annotation Processing Script
=========================================

This script processes video files and their corresponding XML annotations to extract frames and generate labeled data for fish species.
It saves the processed frames and annotations for further analysis.

Author: ahsanjalal, Jan M. Straub
Date: 2024-08-12

Constants:
- VID_DIR: Directory containing the input video files.
- XML_DIR: Directory containing the XML annotation files.
- SAVE_IMG_DIR: Directory to save processed frames.
- SAVE_LAB_DIR: Directory to save annotation files.
- SPECIE_LIST: List of fish species to be labeled.
- OTHER_CLASS: Label for other fish species not in SPECIE_LIST.
- OTHER_LABEL: Numeric label for other fish species.

Functions:
- process_video: Processes a single video file and its corresponding XML annotation.
- process_frame: Processes a single frame and extracts fish annotations.
- main: Main function to process all video files in the VID_DIR directory.

Usage:
- Run this script to process all videos in the VID_DIR and save the results in SAVE_IMG_DIR and SAVE_LAB_DIR.
"""

import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import imageio.v3 as iio

# Constants
VID_DIR = "/Users/jan/Documents/code/cv/project/data/fishclef_2015_release/training_set/videos"
XML_DIR = "/Users/jan/Documents/code/cv/project/data/fishclef_2015_release/training_set/gt"
SAVE_IMG_DIR = "/Users/jan/Documents/code/cv/project/training/img_pool_retrain"
SAVE_LAB_DIR = SAVE_IMG_DIR
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
]
OTHER_CLASS = "others"
OTHER_LABEL = 15


def process_video(video_path, xml_path, img_counter):
    """
    Processes a single video file and its corresponding XML annotation.

    Parameters:
    - video_path: Path to the video file.
    - xml_path: Path to the XML annotation file.
    - img_counter: Counter for the number of processed images.

    Returns:
    - img_counter: Updated counter for the number of processed images.
    - other_fish_count: Count of fish labeled as 'others'.
    """
    cap = cv2.VideoCapture(video_path)
    image_vid = []
    success, image = cap.read()
    while success:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert image to RGB
        image_vid.append(image)  # Append frame to the list
        success, image = cap.read()
    cap.release()

    tree = ET.parse(xml_path)  # Parse the XML file
    root = tree.getroot()
    vid_name_short = os.path.basename(video_path).split(".")[0][-15:]

    other_fish_count = 0
    for child in root:
        frame_id = int(child.attrib["id"])
        if frame_id < len(image_vid):
            process_frame(child, frame_id, image_vid, vid_name_short, img_counter)
            img_counter += 1

    return img_counter, other_fish_count


def process_frame(child, frame_id, image_vid, vid_name_short, img_counter):
    """
    Processes a single frame and extracts fish annotations.

    Parameters:
    - child: XML element containing frame annotations.
    - frame_id: ID of the frame.
    - image_vid: List of frames from the video.
    - vid_name_short: Shortened video name for file naming.
    - img_counter: Counter for the number of processed images.
    """
    for g_child in child:
        fish_specie = g_child.attrib["fish_species"].lower()
        if fish_specie == "chaetodon lununatus":
            fish_specie = "chaetodon lunulatus"  # Correct species name

        # Extract bounding box coordinates
        x, y, w, h = map(
            int,
            [
                g_child.attrib["x"],
                g_child.attrib["y"],
                g_child.attrib["w"],
                g_child.attrib["h"],
            ],
        )
        x, y = max(x, 0), max(y, 0)  # Ensure coordinates are non-negative
        frame = image_vid[frame_id]
        frame = cv2.resize(frame, [640, 640])  # Resize frame
        height, width, _ = frame.shape

        # Normalize bounding box coordinates
        mid_x, mid_y = (x + x + w) / (2 * width), (y + y + h) / (2 * height)
        box_width, box_height = w / width, h / height
        filename = f"{vid_name_short}_image_{frame_id}"
        iio.imwrite(f"{SAVE_IMG_DIR}{filename}.jpg", frame)  # Save frame

        # Determine fish label
        fish_lab = (
            SPECIE_LIST.index(fish_specie)
            if fish_specie in SPECIE_LIST
            else OTHER_LABEL
        )
        item = f"{fish_lab} {mid_x} {mid_y} {box_width} {box_height}"
        with open(f"{SAVE_LAB_DIR}{filename}.txt", "a") as a:
            a.write(item + "\n")  # Save annotation


def main():
    """
    Main function to process all video files in the VID_DIR directory.

    This function performs the following steps:
    1. Changes the working directory to VID_DIR.
    2. Retrieves a list of video files in the VID_DIR directory.
    3. Initializes an image counter.
    4. Processes each video file and its corresponding XML annotation.
    5. Prints the total count of fish labeled as 'others'.
    """
    os.chdir(VID_DIR)
    sub_list = np.array(os.listdir(VID_DIR))
    img_counter = 0

    for vid_count, video in enumerate(sub_list):
        print(f"video number: {vid_count} is in progress")
        video_path = os.path.join(VID_DIR, video)
        xml_path = os.path.join(XML_DIR, os.path.splitext(video)[0] + ".xml")
        img_counter, other_fish_count = process_video(video_path, xml_path, img_counter)

    print("total count for other fish is:", other_fish_count)


if __name__ == "__main__":
    """
    Entry point of the script.

    This block checks if the script is being run directly (not imported as a module).
    If so, it calls the main() function to start processing the video files.
    """
    main()