import os
import cv2
import glob
import numpy as np
import xml.etree.ElementTree as ET

from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = Path("/Users/jan/Documents/code/cv/project")

# Training set
TRAIN_VIDEO_DIR = BASE_DIR / "data/fishclef_2015_release/training_set/videos"
TRAIN_GT_DIR = BASE_DIR / "data/fishclef_2015_release/training_set/gt"
TRAIN_IMG_DIR = BASE_DIR / "train_img/"
TRAIN_GMM_DIR = BASE_DIR / "train_gmm/"
TRAIN_OPTICAL_DIR = BASE_DIR / "train_optical/"
TRAIN_GMM_OPTICAL_DIR = BASE_DIR / "train_gmm_optical/"

# Test set
TEST_VIDEO_DIR = BASE_DIR / "data/fishclef_2015_release/test_set/videos"
TEST_GT_DIR = BASE_DIR / "data/fishclef_2015_release/test_set/gt"
TEST_IMG_DIR = BASE_DIR / "test_img/"
TEST_GMM_DIR = BASE_DIR / "test_gmm/"
TEST_OPTICAL_DIR = BASE_DIR / "test_optical/"
TEST_GMM_OPTICAL_DIR = BASE_DIR / "test_gmm_optical/"

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

UNKNOWN_LABEL = 15

FOREGROUND_DETECTOR_PARAMS = {
    "history": 250,
    "varThreshold": 16,
    "detectShadows": True,
}
BLOB_ANALYSIS_PARAMS = {"min_area": 200}
STRUCTURING_ELEMENT_OPEN = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
STRUCTURING_ELEMENT_CLOSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
FRAME_RESIZE = (640, 640)
FRAME_ADJUST_GAMMA = 1.5
FARNEBACK_PARAMS = {
    "pyr_scale": 0.5,
    "levels": 3,
    "winsize": 15,
    "iterations": 3,
    "poly_n": 5,
    "poly_sigma": 1.2,
    "flags": 0,
}


def adjust_gamma(image, gamma=1.0):
    """
    Adjust the gamma of an image.

    Parameters:
    - image (numpy.ndarray): The input image on which gamma correction is to be applied.
    - gamma (float): The gamma value for correction. Default is 1.0. Values less than 1.0 will make the image darker,
                     while values greater than 1.0 will make the image lighter.

    Returns:
    - numpy.ndarray: The gamma-corrected image.
    """
    # Calculate the inverse of the gamma value
    invGamma = 1.0 / gamma

    # Build a lookup table mapping pixel values [0, 255] to their adjusted gamma values
    table = np.array([(i / 255.0) ** invGamma * 255 for i in range(256)], dtype="uint8")

    # Apply the gamma correction using the lookup table
    return cv2.LUT(image, table)


def save_gmm_annotation(annotation_filename, bboxes, image_width, image_height):
    """
    Save annotations in YOLO format for Gaussian Mixture Model (GMM) detected bounding boxes.

    Parameters:
    - annotation_filename (str): The file to save annotations.
    - bboxes (list of tuples): List of bounding boxes, where each bounding box is represented as a tuple (x, y, width, height).
    - image_width (int): Width of the image.
    - image_height (int): Height of the image.

    Returns:
    - None
    """
    # Open the annotation file in write mode
    with open(annotation_filename, "w") as file:
        # Iterate over each bounding box
        for x, y, width, height in bboxes:
            # Normalize the coordinates
            x_center = (x + width / 2.0) / image_width
            y_center = (y + height / 2.0) / image_height
            width /= image_width
            height /= image_height

            # Write the normalized coordinates to the file in YOLO format
            file.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


def save_annotation_batch_test(
    name, annotation_file_path, bboxes, image_width, image_height
):
    """
    Save annotations in YOLO format for each frame in batches.

    Parameters:
    - name (str): Base name for the annotation files.
    - annotation_file_path (Path): Path object representing the directory to save annotation files.
    - bboxes (list of dict): List of bounding boxes, where each bounding box is represented as a dictionary with keys 'frame_id', 'fish_species', 'x', 'y', 'w', 'h'.
    - image_width (int): Width of the image.
    - image_height (int): Height of the image.

    Returns:
    - None
    """
    # Group bounding boxes by frame_id
    frame_bboxes = {}
    for bbox in bboxes:
        frame_id = bbox["frame_id"]
        frame_bboxes.setdefault(frame_id, []).append(bbox)

    # Prepare content for each file
    file_contents = {}
    for frame_id, bboxes in frame_bboxes.items():
        # Collect annotation lines for this frame
        content = []
        for fish in bboxes:
            fish_species = fish.get("species_name", "").lower()
            x, y, width, height = (
                fish.get("x", 0),
                fish.get("y", 0),
                fish.get("w", 0),
                fish.get("h", 0),
            )

            # Normalize the coordinates
            x_center = (x + width / 2.0) / image_width
            y_center = (y + height / 2.0) / image_height
            width /= image_width
            height /= image_height

            # Determine the species index
            species_index = (
                SPECIES_LIST.index(fish_species)
                if fish_species in SPECIES_LIST
                else UNKNOWN_LABEL
            )

            # Format the annotation line in YOLO format
            content.append(
                f"{species_index} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            )

        # Add content for the current frame
        file_contents[frame_id] = "\n".join(content)

    # Write all files in a batch
    for frame_id, content in file_contents.items():
        # Create a unique file name for each frame
        frame_annotation_file = annotation_file_path / f"{name}_{frame_id:04d}.txt"

        # Write the content to the file
        with open(frame_annotation_file, "w") as file:
            file.write(content)


def save_annotation_batch_train(
    name, annotation_file_path, bboxes, image_width, image_height
):
    """
    Save annotations in YOLO format for each frame in batches.

    Parameters:
    - name (str): Base name for the annotation files.
    - annotation_file_path (Path): Path object representing the directory to save annotation files.
    - bboxes (list of dict): List of bounding boxes, where each bounding box is represented as a dictionary with keys 'frame_id', 'fish_species', 'x', 'y', 'w', 'h'.
    - image_width (int): Width of the image.
    - image_height (int): Height of the image.

    Returns:
    - None
    """
    # Group bounding boxes by frame_id
    frame_bboxes = {}
    for bbox in bboxes:
        frame_id = bbox["frame_id"]
        frame_bboxes.setdefault(frame_id, []).append(bbox)

    # Prepare content for each file
    file_contents = {}
    for frame_id, bboxes in frame_bboxes.items():
        # Collect annotation lines for this frame
        content = []
        for fish in bboxes:
            fish_species = fish.get("fish_species", "").lower()
            x, y, width, height = (
                fish.get("x", 0),
                fish.get("y", 0),
                fish.get("w", 0),
                fish.get("h", 0),
            )

            # Normalize the coordinates
            x_center = (x + width / 2.0) / image_width
            y_center = (y + height / 2.0) / image_height
            width /= image_width
            height /= image_height

            # Determine the species index
            species_index = (
                SPECIES_LIST.index(fish_species)
                if fish_species in SPECIES_LIST
                else UNKNOWN_LABEL
            )

            # Format the annotation line in YOLO format
            content.append(
                f"{species_index} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            )

        # Add content for the current frame
        file_contents[frame_id] = "\n".join(content)

    # Write all files in a batch
    for frame_id, content in file_contents.items():
        # Create a unique file name for each frame
        frame_annotation_file = annotation_file_path / f"{name}_{frame_id:04d}.txt"

        # Write the content to the file
        with open(frame_annotation_file, "w") as file:
            file.write(content)


def extract_ground_truth(video_path):
    """
    Extract ground truth annotations from an XML file corresponding to a video.

    Parameters:
    - video_path (Path): Path object representing the path to the video file.

    Returns:
    - list of dict: A list of dictionaries, where each dictionary contains the ground truth annotations for a frame.
      Each dictionary has the following keys:
        - frame_id (int): The ID of the frame.
        - fish_species (str): The species of the fish.
        - x (int): The x-coordinate of the bounding box.
        - y (int): The y-coordinate of the bounding box.
        - w (int): The width of the bounding box.
        - h (int): The height of the bounding box.
    """
    # Extract the file name without extension from the video path
    file_name_without_ext = video_path.stem

    # Construct the path to the ground truth XML file
    gt_xml_path = TRAIN_GT_DIR / f"{file_name_without_ext}.xml"

    # Check if the ground truth XML file exists
    if not gt_xml_path.exists():
        print(f"Ground truth XML not found: {gt_xml_path}")
        return []

    # Parse the XML file
    tree = ET.parse(gt_xml_path)
    root = tree.getroot()

    # Initialize an empty list to store ground truth annotations
    ground_truth = []

    # Iterate over each frame element in the XML
    for frame in root.findall("frame"):
        frame_id = int(frame.get("id"))

        # Iterate over each object element within the frame
        for obj in frame.findall("object"):
            # Append the ground truth annotation to the list
            ground_truth.append(
                {
                    "frame_id": frame_id,
                    "fish_species": obj.get("fish_species"),
                    "x": int(obj.get("x")),
                    "y": int(obj.get("y")),
                    "w": int(obj.get("w")),
                    "h": int(obj.get("h")),
                }
            )

    return ground_truth


def apply_gmm(frame, frame_idx, gmm_dir, foreground_detector):
    """
    Apply Gaussian Mixture Model (GMM) to a video frame to detect foreground objects and save the results.

    Parameters:
    - frame (numpy.ndarray): The input video frame.
    - frame_idx (int): The index of the current frame.
    - gmm_dir (Path): Path object representing the directory to save GMM results.
    - foreground_detector (cv2.BackgroundSubtractor): The foreground detector object.

    Returns:
    - numpy.ndarray: The processed foreground mask.
    """
    # Apply the foreground detector to the frame
    foreground = foreground_detector.apply(frame)

    # Apply morphological opening to remove noise
    filtered_foreground = cv2.morphologyEx(
        foreground, cv2.MORPH_OPEN, STRUCTURING_ELEMENT_OPEN
    )

    # Apply morphological closing to fill gaps
    filtered_foreground = cv2.morphologyEx(
        filtered_foreground, cv2.MORPH_CLOSE, STRUCTURING_ELEMENT_CLOSE
    )

    # Find contours in the filtered foreground mask
    contours, _ = cv2.findContours(
        filtered_foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Filter contours based on minimum area and compute bounding boxes
    bboxes = [
        cv2.boundingRect(c)
        for c in contours
        if cv2.contourArea(c) >= BLOB_ANALYSIS_PARAMS["min_area"]
    ]

    # Save the filtered foreground mask as an image
    gmm_frame_path = gmm_dir / f"gmm_img_{frame_idx:04d}.jpg"
    cv2.imwrite(str(gmm_frame_path), filtered_foreground)

    # Save the bounding boxes as annotations
    gmm_annotation_path = gmm_dir / f"gmm_img_{frame_idx:04d}.txt"
    if bboxes:
        save_gmm_annotation(
            gmm_annotation_path, bboxes, FRAME_RESIZE[0], FRAME_RESIZE[1]
        )
    else:
        # Create an empty annotation file if no bounding boxes are found
        gmm_annotation_path.touch()

    return filtered_foreground


def apply_optical_flow(frame, frame_idx, prvs, hsv, flow_dir):
    """
    Apply Farneback optical flow to a video frame and save the results.

    Parameters:
    - frame (numpy.ndarray): The current video frame.
    - frame_idx (int): The index of the current frame.
    - prvs (numpy.ndarray): The previous grayscale frame.
    - hsv (numpy.ndarray): The HSV image used for visualizing the optical flow.
    - flow_dir (Path): Path object representing the directory to save optical flow results.

    Returns:
    - tuple: A tuple containing:
        - bgr_resized (numpy.ndarray): The resized BGR image representing the optical flow.
        - next_frame (numpy.ndarray): The next grayscale frame.
    """
    # Convert the current frame to grayscale
    next_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate the optical flow using Farneback method
    flow = cv2.calcOpticalFlowFarneback(prvs, next_frame, None, **FARNEBACK_PARAMS)

    # Compute the magnitude and angle of the flow
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Set the hue of the HSV image based on the angle of the flow
    hsv[..., 0] = ang * 180 / np.pi / 2

    # Set the value of the HSV image based on the normalized magnitude of the flow
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    # Convert the HSV image to BGR for visualization
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Resize the BGR image to match the desired frame size (640x640)
    bgr_resized = cv2.resize(bgr, FRAME_RESIZE)

    # Construct the path to save the optical flow frame
    flow_frame_path = flow_dir / f"flow_img_{frame_idx:04d}.jpg"

    # Save the resized BGR image to the specified directory
    cv2.imwrite(str(flow_frame_path), bgr_resized)

    return bgr_resized, next_frame


def apply_combination(
    frame, frame_idx, filtered_foreground, bgr_resized, gt_bboxes, combined_dir
):
    """
    Combine GMM and Optical Flow images, save the combined image, and save ground truth annotations.

    Parameters:
    - frame (numpy.ndarray): The original video frame.
    - frame_idx (int): The index of the current frame.
    - filtered_foreground (numpy.ndarray): The foreground mask obtained from GMM.
    - bgr_resized (numpy.ndarray): The resized BGR image obtained from optical flow.
    - gt_bboxes (list of dict): List of ground truth bounding boxes for the frame.
    - combined_dir (Path): Path object representing the directory to save combined results.

    Returns:
    - None
    """
    # Initialize a blank image with the same shape as the original frame
    combined_frame = np.zeros_like(frame)

    # Combine the filtered foreground mask into the green channel
    combined_frame[:, :, 1] = filtered_foreground

    # Combine the blue channel of the resized BGR image into the red channel
    combined_frame[:, :, 2] = bgr_resized[:, :, 0]  # Use resized bgr

    # Construct the path to save the combined image
    combined_frame_path = combined_dir / f"combined_img_{frame_idx:04d}.jpg"

    # Save the combined image to the specified directory
    cv2.imwrite(str(combined_frame_path), combined_frame)

    # Construct the path to save the ground truth annotations
    combined_annotation_path = combined_dir / f"combined_img_{frame_idx:04d}.txt"
    name = "combined_img"

    # Save the ground truth annotations if they exist
    if gt_bboxes:
        if "train" in str(combined_dir):
            save_annotation_batch_train(
                name,
                combined_dir,
                gt_bboxes,
                FRAME_RESIZE[0],
                FRAME_RESIZE[1],
            )
        
        if "test" in str(combined_dir):
            save_annotation_batch_test(
                name,
                combined_dir,
                gt_bboxes,
                FRAME_RESIZE[0],
                FRAME_RESIZE[1],
            )

    else:
        # Create an empty annotation file if no ground truth bounding boxes are found
        combined_annotation_path.touch()


def process_frame(
    frame,
    frame1,
    frame_idx,
    gt_bboxes,
    foreground_detector,
    prvs,
    hsv,
    img_dir,
    gmm_dir,
    flow_dir,
    combined_dir,
):
    """
    Process a video frame by applying GMM and Optical Flow, and save the results.

    Parameters:
    - frame (numpy.ndarray): The current video frame.
    - frame1 (numpy.ndarray): The next video frame for optical flow calculation.
    - frame_idx (int): The index of the current frame.
    - gt_bboxes (list of dict): List of ground truth bounding boxes for the frame.
    - foreground_detector (cv2.BackgroundSubtractor): The foreground detector object.
    - prvs (numpy.ndarray): The previous grayscale frame for optical flow calculation.
    - hsv (numpy.ndarray): The HSV image used for visualizing the optical flow.
    - img_dir (Path): Path object representing the directory to save original frames.
    - gmm_dir (Path): Path object representing the directory to save GMM results.
    - flow_dir (Path): Path object representing the directory to save optical flow results.
    - combined_dir (Path): Path object representing the directory to save combined results.

    Returns:
    - numpy.ndarray: The next grayscale frame for optical flow calculation.
    """
    # Save the original frame to the img_dir
    img_frame_path = img_dir / f"img_{frame_idx:04d}.jpg"
    cv2.imwrite(str(img_frame_path), frame)

    # Save annotations for the original frame (train_img)
    img_annotation_path = img_dir / f"img_{frame_idx:04d}.txt"
    name = "img"
    if gt_bboxes:
        # Save ground truth annotations if they exist
        if "train" in str(img_dir):
            save_annotation_batch_train(
                name,
                img_dir,
                gt_bboxes,
                FRAME_RESIZE[0],
                FRAME_RESIZE[1],
            )
        
        if "test" in str(img_dir):
            save_annotation_batch_test(
                name,
                img_dir,
                gt_bboxes,
                FRAME_RESIZE[0],
                FRAME_RESIZE[1],
            )
    else:
        # Create an empty annotation file if no ground truth bounding boxes are found
        img_annotation_path.touch()

    # Apply GMM to the frame to detect foreground objects
    foreground = apply_gmm(frame, frame_idx, gmm_dir, foreground_detector)

    # Apply optical flow to the next frame
    bgr, next_frame = apply_optical_flow(frame1, frame_idx, prvs, hsv, flow_dir)

    # Combine GMM and optical flow results and save the combined image
    apply_combination(frame, frame_idx, foreground, bgr, gt_bboxes, combined_dir)

    return next_frame


def process_video(video_path):
    """
    Process a video to extract ground truth, apply GMM and optical flow, and save the results.

    Parameters:
    - video_path (Path): Path object representing the path to the video file.

    Returns:
    - None
    """
    # Extract the last 15 characters of the video file name (without extension) to use as a directory name
    video_name_short = video_path.stem[-15:]

    # Define directories for saving images, GMM results, optical flow results, and combined results
    img_dir = TRAIN_IMG_DIR / video_name_short
    gmm_dir = TRAIN_GMM_DIR / video_name_short
    flow_dir = TRAIN_OPTICAL_DIR / video_name_short
    combined_dir = TRAIN_GMM_OPTICAL_DIR / video_name_short

    # Create the directories if they do not exist
    for directory in [img_dir, gmm_dir, flow_dir, combined_dir]:
        os.makedirs(directory, exist_ok=True)

    # Extract ground truth bounding boxes from the corresponding XML file
    gt_bboxes = extract_ground_truth(video_path)

    # Open the video file
    cap = cv2.VideoCapture(str(video_path))

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create a foreground detector using MOG2
    foreground_detector = cv2.createBackgroundSubtractorMOG2(
        **FOREGROUND_DETECTOR_PARAMS
    )

    # Read the first frame of the video
    ret, frame1 = cap.read()

    # Check if the video file was read successfully
    if not ret:
        print(f"Failed to read the video file: {video_path}")
        return

    # Convert the first frame to grayscale for optical flow calculation
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    # Initialize an HSV image for visualizing optical flow
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255

    # Initialize the frame index
    frame_idx = 0

    # Process each frame of the video
    with tqdm(total=total_frames, desc=f"Processing {video_name_short}") as video_pbar:
        while ret:
            # Resize the frame and adjust its gamma
            frame = cv2.resize(frame1, FRAME_RESIZE)
            frame = adjust_gamma(frame, FRAME_ADJUST_GAMMA)

            # Process the current frame
            next_frame = process_frame(
                frame,
                frame1,
                frame_idx,
                gt_bboxes,
                foreground_detector,
                prvs,
                hsv,
                img_dir,
                gmm_dir,
                flow_dir,
                combined_dir,
            )

            # Update the progress bar
            video_pbar.update(1)

            # Update the previous frame for optical flow calculation
            prvs = next_frame

            # Read the next frame of the video
            ret, frame1 = cap.read()

            # Increment the frame index
            frame_idx += 1

    # Release the video capture object
    cap.release()


def main():
    """
    Main function to process all video files in the specified directory.

    This function searches for video files with .flv and .avi extensions in the VIDEO_DIR,
    and processes each video file using the process_video function.
    """
    # Get a list of all .flv and .avi video files in the VIDEO_DIR
    video_files = list(TRAIN_VIDEO_DIR.glob("*.flv")) + list(TRAIN_VIDEO_DIR.glob("*.avi"))

    # Use ThreadPoolExecutor to process videos concurrently
    with ThreadPoolExecutor() as executor:
        # Submit all video processing tasks to the thread pool
        futures = [executor.submit(process_video, video) for video in video_files]

        # Optionally, wait for all the futures to complete and handle any exceptions
        for future in as_completed(futures):
            try:
                future.result()  # Retrieve the result of the function (if any)
            except Exception as exc:
                print(f"An error occurred: {exc}")

if __name__ == "__main__":
    # Entry point of the script
    main()