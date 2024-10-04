import os
import cv2
import numpy as np

from tqdm import tqdm

from helper import adjust_gamma, get_annotation, extract_ground_truth


def apply_gmm(frame, foreground_detector):
    """
    Applies GMM (Gaussian Mixture Model) to detect foreground objects in a frame.

    Args:
        frame (np.ndarray): Input frame.
        foreground_detector (cv2.BackgroundSubtractorMOG2): Foreground detector.

    Returns:
        np.ndarray: Filtered foreground mask.
    """
    foreground = foreground_detector.apply(frame)
    filtered_foreground = cv2.morphologyEx(
        foreground, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )
    filtered_foreground = cv2.morphologyEx(
        filtered_foreground,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )

    # Shadow Removal: Convert shadows to binary foreground
    _, filtered_foreground = cv2.threshold(
        filtered_foreground, 127, 255, cv2.THRESH_BINARY
    )

    return filtered_foreground


def apply_optical_flow(frame, prvs, hsv, farneback, frame_size):
    """
    Computes optical flow using Farneback method and visualizes it in HSV space.

    Args:
        frame (np.ndarray): Input frame.
        prvs (np.ndarray): Previous frame in grayscale.
        hsv (np.ndarray): HSV image used for optical flow visualization.
        farneback (dict): Parameters for the Farneback optical flow algorithm.
        frame_size (tuple): Resized image size.

    Returns:
        tuple: Tuple containing resized BGR image of the flow and next grayscale frame.
    """
    next_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prvs, next_frame, None, **farneback)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    bgr_resized = cv2.resize(bgr, frame_size)

    return bgr_resized, next_frame


def apply_combination(
    frame,
    frame_idx,
    filtered_foreground,
    bgr_resized,
    gt_bboxes,
    image_width,
    image_height,
    combined_dir,
    species_list,
    unknown_label,
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
        image_width (int): Width of the image.
        image_height (int): Height of the image.
        combined_dir (Path): Directory to save the combined image and annotations.
        species_list (list): List of species from SPECIES_LIST.
        unknown_label (int): Label for unknown fish species.
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
            species_list,
            unknown_label,
            image_width,
            image_height,
        )


def process_frame(
    frame,
    frame1,
    frame_idx,
    gt_bboxes,
    foreground_detector,
    prvs,
    hsv,
    farneback,
    frame_size,
    img_dir,
    combined_dir,
    save_original,
    species_list,
    unknown_label,
    opacity_foreground,
    opacity_optical_flow,
):
    """
    Processes a single video frame by applying background subtraction (GMM) and optical flow,
    and then combines the results. Optionally saves the original frame, and stores the combined
    output along with ground truth annotations.

    Args:
        frame (numpy.ndarray): The current video frame after resizing and gamma adjustment.
        frame1 (numpy.ndarray): The next video frame to compute optical flow.
        frame_idx (int): The index of the current frame in the video.
        gt_bboxes (list): Ground truth bounding boxes for objects (fish) in the frame.
        foreground_detector (cv2.BackgroundSubtractor): Foreground detector based on GMM.
        prvs (numpy.ndarray): The previous grayscale frame used for optical flow calculation.
        hsv (numpy.ndarray): The HSV image used for visualizing optical flow.
        farneback (dict): Parameters for the Farneback optical flow algorithm.
        frame_size (tuple): Resized image size
        img_dir (Path): Directory to save the original frames.
        combined_dir (Path): Directory to save the combined results of GMM and optical flow.
        save_original (bool): True if original ground truth images should be saved.
        species_list (list): List of species from SPECIES_LIST.
        unknown_label (int): Label for unknown fish species.
        opacity_foreground (float): Opacity for filtered foreground mask (0 to 1).
        opacity_optical_flow (float): Opacity for optical flow visualization (0 to 1).

    Returns:
        next_frame (numpy.ndarray): The grayscale version of the current frame (frame1) for use in the next iteration of optical flow calculation.
    """
    if save_original:
        # Save the original frame to the img_dir
        img_frame_path = img_dir / f"img_{frame_idx:04d}.png"
        cv2.imwrite(str(img_frame_path), frame)

    # Apply GMM to the frame to detect foreground objects
    foreground = apply_gmm(frame, foreground_detector)

    # Apply optical flow to the next frame
    bgr, next_frame = apply_optical_flow(frame1, prvs, hsv, farneback, frame_size)

    # Combine GMM and optical flow results and save the combined image
    apply_combination(
        frame,
        frame_idx,
        foreground,
        bgr,
        gt_bboxes,
        frame_size[0],
        frame_size[1],
        combined_dir,
        species_list,
        unknown_label,
        opacity_foreground,
        opacity_optical_flow,
    )

    return next_frame


def process_video(
    video_path,
    farneback,
    frame_size,
    img_dir,
    gt_dir,
    combined_dir,
    save_original,
    species_list,
    unknown_label,
    opacity_foreground,
    opacity_optical_flow,
):
    """
    Processes a video by applying background subtraction (using Gaussian Mixture Model), optical flow, and frame adjustments, and saves the processed frames and combined results along with ground truth annotations.

    Args:
        video_path (Path): Path to the input video file.
        farneback (dict): Parameters for the Farneback optical flow algorithm.
        frame_size (tuple): Resized image size
        img_dir (Path): Path to the video frames.
        gt_dir (Path): Path to the ground truth file.
        combined_dir (Path): Path where the combined GMM-optical flow image is saved.
        save_original (bool): True if original ground truth images should be saved.
        species_list (list): List of species from SPECIES_LIST.
        unknown_label (int): Label for unknown fish species.
        opacity_foreground (float): Opacity for filtered foreground mask (0 to 1).
        opacity_optical_flow (float): Opacity for optical flow visualization (0 to 1).

    Parameters:
        video_path (Path): Path to the video file being processed.

    Returns:
        None: The function processes the video, saves results, and does not return anything.
    """

    video_name_short = video_path.stem[-15:]
    img_dir = img_dir / video_name_short
    combined_dir = combined_dir / video_name_short

    for directory in [combined_dir]:
        os.makedirs(directory, exist_ok=True)

    if save_original:
        for directory in [img_dir]:
            os.makedirs(directory, exist_ok=True)

    # Consider different GT names
    species_key = ""
    if "train" in str(combined_dir):
        species_key = "fish_species"
    if "val" in str(combined_dir):
        species_key = "species_name"

    # Extract ground truth bounding boxes from the corresponding XML file
    gt_bboxes = extract_ground_truth(video_path, species_key, gt_dir)

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
            frame = cv2.resize(frame1, frame_size)
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
                farneback,
                frame_size,
                img_dir,
                combined_dir,
                save_original,
                species_list,
                unknown_label,
                opacity_foreground,
                opacity_optical_flow,
            )

            video_pbar.update(1)
            prvs = next_frame
            ret, frame1 = cap.read()
            frame_idx += 1

    cap.release()
