from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from video_processing import process_video

# Configuration Flags
SAVE_ORIGINAL = False  # Flag to save original frames
RELEASE = False  # Flag to switch between concurrent and sequential processing
TRAIN = True  # Flag to switch between creating training images and creating validation images

# Base directory setup
BASE_DIR = Path("/Users/jan/Documents/code/cv/project")

# Training set directories
TRAIN_VIDEO_DIR = BASE_DIR / "data/fishclef_2015_release/training_set/videos"
TRAIN_GT_DIR = BASE_DIR / "data/fishclef_2015_release/training_set/gt"
TRAIN_IMG_DIR = BASE_DIR / "train_img/"
TRAIN_GMM_OPTICAL_DIR = BASE_DIR / "train_combined/"

# Validation set directories
VAL_VIDEO_DIR = BASE_DIR / "data/fishclef_2015_release/test_set/videos"
VAL_GT_DIR = BASE_DIR / "data/fishclef_2015_release/test_set/gt"
VAL_IMG_DIR = BASE_DIR / "val_img/"
VAL_GMM_OPTICAL_DIR = BASE_DIR / "val_combined/"

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
UNKNOWN_LABEL = len(SPECIES_LIST)

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

# Opacity parameters
OPACITY_FOREGROUND = 0.5
OPACITY_OPTICAL_FLOW = 0.5


def main():
    """
    Main entry point of the script. Processes either training or test videos.
    """

    # Set directories based on whether we are in training or testing mode
    video_dir = TRAIN_VIDEO_DIR if TRAIN else VAL_VIDEO_DIR
    img_dir = TRAIN_IMG_DIR if TRAIN else VAL_IMG_DIR
    gt_dir = TRAIN_GT_DIR if TRAIN else VAL_GT_DIR
    combined_dir = TRAIN_GMM_OPTICAL_DIR if TRAIN else VAL_GMM_OPTICAL_DIR

    video_files = list(video_dir.glob("*.flv"))

    if RELEASE:
        for video in video_files[:1]:
            process_video(
                video,
                FARNEBACK_PARAMS,
                FRAME_RESIZE,
                img_dir,
                gt_dir,
                combined_dir,
                SAVE_ORIGINAL,
                SPECIES_LIST,
                UNKNOWN_LABEL,
                OPACITY_FOREGROUND,
                OPACITY_OPTICAL_FLOW,
            )
    else:
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    process_video,
                    video,
                    FARNEBACK_PARAMS,
                    FRAME_RESIZE,
                    img_dir,
                    gt_dir,
                    combined_dir,
                    SAVE_ORIGINAL,
                    SPECIES_LIST,
                    UNKNOWN_LABEL,
                    OPACITY_FOREGROUND,
                    OPACITY_OPTICAL_FLOW,
                )
                for video in video_files
            ]

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    print(f"An error occurred: {exc}")


if __name__ == "__main__":
    main()
