"""
Fish Species Annotation Processing Script
=========================================

This script processes video files and their corresponding XML annotations to extract frames and generate labeled data for fish species.
It saves the processed frames and annotations for further analysis.

Author: ahsanjalal, JanMStraub
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

import cv2
import xml.etree.ElementTree as ET
import imageio.v3 as iio
from pathlib import Path

BASE_DIR = Path("/Users/jan/Documents/code/cv/project")
VIDEO_DIR = BASE_DIR / "data/fishclef_2015_release/training_set/videos"
XML_DIR = BASE_DIR / "data/fishclef_2015_release/training_set/gt"
IMG_DIR = BASE_DIR / "train_img/"
GMM_DIR = BASE_DIR / "train_gmm/"
OPTICAL_DIR = BASE_DIR / "train_optical/"
GMM_OPTICAL_DIR = BASE_DIR / "train_gmm_optical/"

SPECIES_LIST = [
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

OTHER_LABEL = 15


def process_video_img(video_path, xml_path, img_counter):
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
    cap = cv2.VideoCapture(str(video_path))
    frames = []

    while True:
        success, frame = cap.read()
        if not success:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()

    tree = ET.parse(xml_path)
    root = tree.getroot()
    vid_name_short = video_path.stem[-15:]

    for child in root:
        frame_id = int(child.attrib["id"])
        if frame_id < len(frames):
            img_counter += process_frame(child, frame_id, frames, vid_name_short)

    return img_counter


def process_frame(child, frame_id, frames, vid_name_short):
    """
    Processes a single frame and extracts fish annotations.

    Parameters:
    - child: XML element containing frame annotations.
    - frame_id: ID of the frame.
    - frames: List of frames from the video.
    - vid_name_short: Shortened video name for file naming.
    - img_counter: Counter for the number of processed images.

    Returns:
    - int: Updated image counter after processing the frame.
    """
    frame = cv2.resize(frames[frame_id], (640, 640))
    height, width, _ = frame.shape
    filename = f"{vid_name_short}_image_{frame_id}"
    iio.imwrite(IMG_DIR / f"{filename}.jpg", frame)

    for g_child in child:
        fish_species = g_child.attrib["fish_species"].lower()
        if fish_species == "chaetodon lununatus":
            fish_species = "chaetodon lunulatus"

        x, y, w, h = (int(g_child.attrib[k]) for k in ("x", "y", "w", "h"))
        x, y = max(x, 0), max(y, 0)
        mid_x, mid_y = (x + w / 2) / width, (y + h / 2) / height
        box_width, box_height = w / width, h / height

        fish_label = SPECIES_LIST.index(fish_species) if fish_species in SPECIES_LIST else OTHER_LABEL
        annotation = f"{fish_label} {mid_x:.6f} {mid_y:.6f} {box_width:.6f} {box_height:.6f}"

        with open(IMG_DIR / f"{filename}.txt", "a") as label_file:
            label_file.write(annotation + "\n")

    return 1


def main():
    """
    Main function to process all video files in the VIDEO_DIR directory.
    """
    # Get all .flv and .avi files in the video directory
    video_files = list(VIDEO_DIR.glob("*.flv")) + list(VIDEO_DIR.glob("*.avi"))
    img_counter = 0

    for vid_count, video_path in enumerate(video_files):
        print(f"Processing video {vid_count + 1}/{len(video_files)}: {video_path.name}")
        
        # Create the xml_path as a Path object
        xml_path = XML_DIR / (video_path.stem + ".xml")
        
        # Process the video and update counters
        img_counter = process_video_img(video_path, xml_path, img_counter)

    print(f"{img_counter} images where processed.")


if __name__ == "__main__":
    """
    Entry point of the script.

    This block checks if the script is being run directly (not imported as a module).
    If so, it calls the main() function to start processing the video files.
    """
    main()