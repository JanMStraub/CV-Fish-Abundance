import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET

from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration Flags
SAVE_ORIGINAL = False  # Flag to save original frames
RELEASE = False  # Flag to switch between concurrent and sequential processing

# Base directory setup
BASE_DIR = Path("/home/jan/Documents/code/CV-Fish-Abundance")

# Training set directories
TRAIN_VIDEO_DIR = BASE_DIR / "data/fishclef_2015_release/training_set/videos"
TRAIN_GT_DIR = BASE_DIR / "data/fishclef_2015_release/training_set/gt"
TRAIN_IMG_DIR = BASE_DIR / "train_img/"
TRAIN_GMM_DIR = BASE_DIR / "train_gmm/"
TRAIN_OPTICAL_DIR = BASE_DIR / "train_optical/"
TRAIN_GMM_OPTICAL_DIR = BASE_DIR / "train_gmm_optical/"

# Test set directories
TEST_VIDEO_DIR = BASE_DIR / "data/fishclef_2015_release/test_set/videos"
TEST_GT_DIR = BASE_DIR / "data/fishclef_2015_release/test_set/gt"
TEST_IMG_DIR = BASE_DIR / "test_img/"
TEST_GMM_DIR = BASE_DIR / "test_gmm/"
TEST_OPTICAL_DIR = BASE_DIR / "test_optical/"
TEST_GMM_OPTICAL_DIR = BASE_DIR / "test_gmm_optical/"

# List of species names
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

# Label for unknown species
UNKNOWN_LABEL = 15

# Frame processing parameters
FRAME_RESIZE = (640, 640)

# Optical flow parameters
FARNEBACK_PARAMS = {
    "pyr_scale": 0.95,
    "levels": 10,
    "winsize": 15,
    "iterations": 3,
    "poly_n": 5,
    "poly_sigma": 1.2,
    "flags": 0,
}


def adjust_gamma(image, gamma=1.0):
    """
    Adjusts the gamma of an image.

    Args:
        image (np.ndarray): Input image.
        gamma (float): Gamma value to adjust (default is 1.0).

    Returns:
        np.ndarray: Gamma adjusted image.
    """
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in range(256)], dtype="uint8")
    return cv2.LUT(image, table)


def get_annotation(
    name,
    annotation_file_path,
    bboxes,
    image_width,
    image_height,
):
    """
    Generates YOLO format annotations for bounding boxes and saves them to files.

    Args:
        name (str): Name prefix for saved annotation files.
        annotation_file_path (Path): Path where annotation files will be saved.
        bboxes (list): List of bounding boxes for the frame.
        image_width (int): Width of the image.
        image_height (int): Height of the image.
    """
    frame_bboxes = {}

    for bbox in bboxes:
        frame_id = bbox["frame_id"]
        frame_bboxes.setdefault(frame_id, []).append(bbox)

    for frame_id, bboxes in frame_bboxes.items():
        annotations = []
        for fish in bboxes:
            fish_species = fish.get("fish_species", "").lower()
            x, y, width, height = (
                fish.get("x", 0),
                fish.get("y", 0),
                fish.get("w", 0),
                fish.get("h", 0),
            )
            x_center = (x + width / 2.0) / image_width
            y_center = (y + height / 2.0) / image_height
            width /= image_width
            height /= image_height
            species_index = (
                SPECIES_LIST.index(fish_species)
                if fish_species in SPECIES_LIST
                else UNKNOWN_LABEL
            )
            annotations.append(
                f"{species_index} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            )

        frame_annotation_file = annotation_file_path / f"{name}_{frame_id:04d}.txt"
        with open(frame_annotation_file, "w") as file:
            file.write("\n".join(annotations))


def extract_ground_truth(video_path, species_key):
    """
    Extracts ground truth annotations from the corresponding XML file.

    Args:
        video_path (Path): Path to the video file.
        species_key (str): Key for accessing species name in bbox dictionary (default is 'fish_species').

    Returns:
        list: List of ground truth bounding boxes extracted from XML.
    """
    file_name_without_ext = video_path.stem
    gt_xml_path = TEST_GT_DIR / f"{file_name_without_ext}.xml"

    if not gt_xml_path.exists():
        print(f"Ground truth XML not found: {gt_xml_path}")
        return []

    tree = ET.parse(gt_xml_path)
    root = tree.getroot()

    ground_truth = []
    for frame in root.findall("frame"):
        frame_id = int(frame.get("id"))
        for obj in frame.findall("object"):
            ground_truth.append(
                {
                    "frame_id": frame_id,
                    "fish_species": obj.get(species_key),
                    "x": int(obj.get("x")),
                    "y": int(obj.get("y")),
                    "w": int(obj.get("w")),
                    "h": int(obj.get("h")),
                }
            )

    return ground_truth


def apply_gmm(frame, foreground_detector):
    """
    Applies GMM (Gaussian Mixture Model) to detect foreground objects in a frame.

    Args:
        frame (np.ndarray): Input frame.
        foreground_detector (cv2.BackgroundSubtractorMOG2): Foreground detector.

    Returns:
        np.ndarray: Filtered foreground mask.
    """
    # frame_denoised = cv2.fastNlMeansDenoising(frame, None)
    foreground = foreground_detector.apply(frame)
    filtered_foreground = cv2.morphologyEx(
        foreground, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )
    filtered_foreground = cv2.morphologyEx(
        filtered_foreground,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    # frame_denoised = cv2.fastNlMeansDenoising(filtered_foreground, None)

    # Shadow Removal: Convert shadows to binary foreground
    _, filtered_foreground = cv2.threshold(
        filtered_foreground, 127, 255, cv2.THRESH_BINARY
    )

    return filtered_foreground


def apply_optical_flow(frame, prvs, hsv):
    """
    Computes optical flow using Farneback method and visualizes it in HSV space.

    Args:
        frame (np.ndarray): Input frame.
        prvs (np.ndarray): Previous frame in grayscale.
        hsv (np.ndarray): HSV image used for optical flow visualization.

    Returns:
        tuple: Tuple containing resized BGR image of the flow and next grayscale frame.
    """
    next_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prvs, next_frame, None, **FARNEBACK_PARAMS)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    bgr_resized = cv2.resize(bgr, FRAME_RESIZE)

    return bgr_resized, next_frame


def apply_combination(
    frame,
    frame_idx,
    filtered_foreground,
    bgr_resized,
    gt_bboxes,
    combined_dir,
    opacity_foreground=0.5,
    opacity_optical_flow=0.5,
):
    """
    Combines the results of GMM and optical flow with opacity blending, and saves the combined image and annotations.

    Args:
        frame (np.ndarray): Original frame.
        frame_idx (int): Frame index.
        filtered_foreground (np.ndarray): Foreground mask obtained from GMM.
        bgr_resized (np.ndarray): Optical flow visualization in BGR format.
        gt_bboxes (list): List of ground truth bounding boxes.
        combined_dir (Path): Directory to save the combined image and annotations.
        opacity_foreground (float): Opacity for filtered foreground mask (0 to 1).
        opacity_optical_flow (float): Opacity for optical flow visualization (0 to 1).
    """

    combined_frame = np.zeros_like(frame)
    grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    filtered_foreground_normalized = cv2.normalize(
        filtered_foreground, None, 0, 255, cv2.NORM_MINMAX
    )
    blended_foreground = cv2.addWeighted(
        grayscale_frame,
        1 - opacity_foreground,
        filtered_foreground_normalized,
        opacity_foreground,
        0,
    )
    blue_channel_optical_flow = bgr_resized[:, :, 0]
    blended_optical_flow = cv2.addWeighted(
        grayscale_frame,
        1 - opacity_optical_flow,
        blue_channel_optical_flow,
        opacity_optical_flow,
        0,
    )

    combined_frame[:, :, 0] = grayscale_frame  # Grayscale frame
    combined_frame[:, :, 1] = blended_foreground  # filtered foreground
    combined_frame[:, :, 2] = blended_optical_flow  # Blended optical flow

    combined_frame_path = combined_dir / f"combined_img_{frame_idx:04d}.jpg"
    cv2.imwrite(str(combined_frame_path), combined_frame)

    if gt_bboxes:
        get_annotation(
            "combined_img",
            combined_dir,
            gt_bboxes,
            FRAME_RESIZE[0],
            FRAME_RESIZE[1],
        )


def process_frame(
    frame,
    frame1,
    frame_idx,
    gt_bboxes,
    foreground_detector,
    prvs,
    hsv,
    img_dir,
    combined_dir,
):
    """
    Processes a single video frame by applying background subtraction (GMM) and optical flow,
    and then combines the results. Optionally saves the original frame, and stores the combined
    output along with ground truth annotations.

    This function performs the following steps for a given frame:

    1. Optionally saves the original frame to a specified directory.
    2. Applies Gaussian Mixture Model (GMM) to detect foreground objects in the frame.
    3. Computes optical flow between the current and next frame to track movement.
    4. Combines the GMM results and optical flow into a final output image.
    5. Saves the combined image and associated ground truth annotations to the specified directory.

    Args:
        frame (numpy.ndarray): The current video frame after resizing and gamma adjustment.
        frame1 (numpy.ndarray): The next video frame to compute optical flow.
        frame_idx (int): The index of the current frame in the video.
        gt_bboxes (list): Ground truth bounding boxes for objects (fish) in the frame.
        foreground_detector (cv2.BackgroundSubtractor): Foreground detector based on GMM.
        prvs (numpy.ndarray): The previous grayscale frame used for optical flow calculation.
        hsv (numpy.ndarray): The HSV image used for visualizing optical flow.
        img_dir (Path): Directory to save the original frames.
        combined_dir (Path): Directory to save the combined results of GMM and optical flow.

    Returns:
        next_frame (numpy.ndarray): The grayscale version of the current frame (frame1) for use in the next iteration of optical flow calculation.
    """
    if SAVE_ORIGINAL:
        # Save the original frame to the img_dir
        img_frame_path = img_dir / f"img_{frame_idx:04d}.png"
        cv2.imwrite(str(img_frame_path), frame)

    # Apply GMM to the frame to detect foreground objects
    foreground = apply_gmm(frame, foreground_detector)

    # Apply optical flow to the next frame
    bgr, next_frame = apply_optical_flow(frame1, prvs, hsv)

    # Combine GMM and optical flow results and save the combined image
    apply_combination(
        frame, frame_idx, foreground, bgr, gt_bboxes, combined_dir
    )

    return next_frame


def process_video(video_path):
    """
    Processes a video by applying background subtraction (using Gaussian Mixture Model),
    optical flow, and frame adjustments, and saves the processed frames and combined results
    along with ground truth annotations.

    This function extracts frames from the input video, performs foreground detection using
    a Gaussian Mixture Model (GMM), calculates optical flow for movement detection, and
    combines these results. The processed frames and combined images are saved in specific
    directories. Additionally, it uses ground truth bounding boxes extracted from an
    XML file for annotation purposes.

    Args:
        video_path (Path): Path to the input video file.

    Steps:
        1. Extract ground truth bounding boxes for the video from the corresponding XML file.
        2. Create directories to store processed images and combined results.
        3. Open the video and initialize background subtraction (GMM) and optical flow.
        4. Process each frame in the video:
            - Resize and adjust gamma for frame.
            - Apply background subtraction (GMM) for foreground detection.
            - Compute optical flow to detect movement.
            - Combine the results of GMM and optical flow.
            - Save the processed frames and results.
        5. Release the video capture object when done.

    The function also includes progress tracking using tqdm to visualize the video processing progress.

    Parameters:
        video_path (Path): Path to the video file being processed.

    Returns:
        None: The function processes the video, saves results, and does not return anything.
    """

    video_name_short = video_path.stem[-15:]
    img_dir = TEST_IMG_DIR / video_name_short
    combined_dir = TEST_GMM_OPTICAL_DIR / video_name_short

    for directory in [combined_dir]:
        os.makedirs(directory, exist_ok=True)

    if SAVE_ORIGINAL:
        for directory in [img_dir]:
            os.makedirs(directory, exist_ok=True)

    # Consider different GT names
    species_key = ""
    if "train" in str(combined_dir):
        species_key = "fish_species"
    if "test" in str(combined_dir):
        species_key = "species_name"

    # Extract ground truth bounding boxes from the corresponding XML file
    gt_bboxes = extract_ground_truth(video_path, species_key)

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    foreground_detector = cv2.createBackgroundSubtractorMOG2(
        history=250, varThreshold=16, detectShadows=True
    )

    ret, frame1 = cap.read()
    if not ret:
        print(f"Failed to read the video file: {video_path}")
        return

    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255
    frame_idx = 0

    # Process each frame of the video
    with tqdm(total=total_frames, desc=f"Processing {video_name_short}") as video_pbar:
        while ret:
            frame = cv2.resize(frame1, FRAME_RESIZE)
            frame = adjust_gamma(frame, 1.5)
            frame_blurred = cv2.GaussianBlur(frame, (5, 5), 0)

            # Process the current frame
            next_frame = process_frame(
                frame_blurred,
                frame1,
                frame_idx,
                gt_bboxes,
                foreground_detector,
                prvs,
                hsv,
                img_dir,
                combined_dir,
            )

            video_pbar.update(1)
            prvs = next_frame
            ret, frame1 = cap.read()
            frame_idx += 1

    cap.release()


def main():
    """
    Main entry point of the script. Processes either training or test videos.
    """
    video_files = list(TEST_VIDEO_DIR.glob("*.flv")) + list(
        TEST_VIDEO_DIR.glob("*.avi")
    )

    if RELEASE:
        for video in video_files[:1]:
            process_video(video)
    else:
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_video, video) for video in video_files]

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    print(f"An error occurred: {exc}")


if __name__ == "__main__":
    main()
