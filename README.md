# Real-Time Underwater Fish Detection and Species Classification Using YOLOv7 Optimized for Google Coral

This repository contains a Python-based implementation developed for the “Computer Vision” course at the University of Heidelberg, Germany. The project focuses on detecting and classifying fish species in underwater videos using background subtraction (GMM) and optical flow as preprocessing steps. The preprocessed frames are then passed to the YOLO model for fish detection and classification.
As a dataset we utilized the [LifeCLEF 2015](https://www.imageclef.org/lifeclef/2015/fish) Fish Dataset for training and evaluation.

This implementation is inspired by the [Fish-Abundance](https://github.com/ahsan856jalal/Fish-Abundance) project by Salman et al.

## Features

- **Gaussian Mixture Model (GMM)** for background subtraction and foreground detection.
- **Optical Flow** (Farneback method) for motion tracking between frames.
- **YOLO-style bounding box annotations** for fish detection based on ground truth data.
- **Concurrent processing** of video files for faster execution.
- Automatic generation of combined images with foreground detection, optical flow, and bounding boxes.
- Configurable parameters such as gamma correction, frame resizing, and visualization opacities.

## Requirements

This project requires Python 3.10 and the following dependencies:

- `opencv-python`
- `numpy`
- `tqdm`
- `imageio`
- `pathlib`

You can install the required dependencies via pip:

```bash
pip install -r requirements.txt
```

## Dataset

This project uses the FishCLEF 2015 dataset. You can download the dataset from the FishCLEF competition website. Once downloaded, the folder structure should be organized as follows:

```bash
.
├── data
│   └── fishclef_2015_release
│       ├── training_set
│       │   ├── videos
│       │   ├── species_samples
│       │   └── gt
│       └── test_set
│           ├── videos
│           └── gt
├── train_img
├── train_combined
├── val_img
└── val_combined
    ├── ...
```

## Usage

### Configuration

Edit the script's configuration flags and parameters at the top of the file to customize the behavior:

- `SAVE_ORIGINAL`: Set to `True` if you want to save the original frames in addition to the processed outputs.
- `RELEASE`: Set to `True` to process a single video (for testing), or `False` to process videos concurrently.
- `TRAIN`: Set to `True` to create training images, or `False` to create validation images.
- `BASE_DIR`: Set the path for the project files.
- `FRAME_RESIZE`: Target frame size for processing (default is `(640, 640)`).
- `FARNEBACK_PARAMS`: Parameters for optical flow computation (can be adjusted based on the scene).
- `OPACITY_FOREGROUND`: Opacity for filtered foreground mask (0 to 1).
- `OPACITY_OPTICAL_FLOW`: Opacity for optical flow visualization (0 to 1).

### Running the Script

To process the videos, simply run the script:

```bash
python main.py
```

This will process all video files in the selected set, extract frames, detect fish using background subtraction and optical flow, and generate combined images along with YOLO-style annotations.

The output will be saved in the following directories:

- `train_img/` or `test_img/`: If `SAVE_ORIGINAL` is enabled, original frames will be saved here.
- `train_combined/` or `test_combined/`: The combined GMM and optical flow results will be saved here, along with ground truth annotations.

### Ground Truth Annotation

The ground truth bounding boxes are extracted from XML files provided in the dataset. The script automatically detects the species and generates YOLO annotations for each frame.

### Parallel Processing

By default, the script uses `ThreadPoolExecutor` to process videos concurrently for faster execution. If you want to process videos sequentially (for debugging or testing or for a real-time application), set `RELEASE` to `True`.

## Directory Structure

The main directories used by the script are:

- **Training Set Directories:**

  - `train_img/`: Original frames from training videos.
  - `train_combined/`: Combined GMM and optical flow results.

- **Test Set Directories:**
  - `val_img/`: Original frames from test videos.
  - `val_combined/`: Combined GMM and optical flow results.

## Example Output

The script generates combined images with the following elements:

- Grayscale background image.
- Foreground detected by GMM (overlayed with adjustable opacity).
- Optical flow visualization (overlayed with adjustable opacity).
- Ground truth bounding boxes in YOLO format.

##

## Contributing

Contributions are welcome! Feel free to open issues or pull requests to improve the functionality, performance, or add new features.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

This project is based on the FishCLEF 2015 dataset, and we thank the organizers of the FishCLEF competition for providing the dataset and ground truth annotations.
