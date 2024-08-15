"""
YOLO on Frames Processing Script
================================

This script processes images using the YOLO object detection model and saves the detection results.
It is designed to work with the FishCLEF or UWA dataset.

Author: ahsanjalal, JanMStraub
Date: 2024-08-12

Constants:
- save_test_part: Directory to save YOLO detection results for the test dataset.
- save_train_part: Directory to save YOLO detection results for the train dataset.

Functions:
- detect: Performs object detection on a given image using the YOLO model.
- main: Main function to process images listed in the validation files and save detection results.

Usage:
- Run this script to process images listed in 'val_from_test.txt' and 'val_from_train.txt' and save the YOLO detection results.
"""

import os
import cv2
import numpy as np
from ctypes import *
from os.path import join
from collections import namedtuple
from pathlib import Path


# Custom structures for YOLO detection
class BOX(Structure):
    _fields_ = [("x", c_float), ("y", c_float), ("w", c_float), ("h", c_float)]


class DETECTION(Structure):
    _fields_ = [
        ("bbox", BOX),
        ("classes", c_int),
        ("prob", POINTER(c_float)),
        ("mask", POINTER(c_float)),
        ("objectness", c_float),
        ("sort_class", c_int),
    ]


class IMAGE(Structure):
    _fields_ = [("w", c_int), ("h", c_int), ("c", c_int), ("data", POINTER(c_float))]


class METADATA(Structure):
    _fields_ = [("classes", c_int), ("names", POINTER(c_char_p))]


# Load darknet library
lib = CDLL(Path(__file__).parent / "../darknet/libdarknet.so", RTLD_GLOBAL)

# Set up function arguments and return types
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA
lib.load_network.argtypes = [c_char_p, c_char_p, c_int]
lib.load_network.restype = c_void_p
lib.load_image_color.argtypes = [c_char_p, c_int, c_int]
lib.load_image_color.restype = IMAGE
lib.network_predict_image.argtypes = [c_void_p, IMAGE]
lib.network_predict_image.restype = POINTER(c_float)
lib.get_network_boxes.argtypes = [
    c_void_p,
    c_int,
    c_int,
    c_float,
    c_float,
    POINTER(c_int),
    c_int,
    POINTER(c_int),
]
lib.get_network_boxes.restype = POINTER(DETECTION)
lib.do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]
lib.free_image.argtypes = [IMAGE]
lib.free_detections.argtypes = [POINTER(DETECTION), c_int]


def detect(net, meta, image, thresh=0.25, hier_thresh=0.5, nms=0.45):
    """
    Performs object detection on a given image using the YOLO model.

    Parameters:
    - net: YOLO network object.
    - meta: Metadata object containing class information.
    - image: Path to the image file.
    - thresh: Detection threshold.
    - hier_thresh: Hierarchical threshold.
    - nms: Non-max suppression threshold.

    Returns:
    - A sorted list of detection results, each containing the class name, probability, and bounding box coordinates.
    """
    im = lib.load_image_color(image.encode("utf-8"), 0, 0)
    num = c_int(0)
    pnum = pointer(num)
    lib.network_predict_image(net, im)
    dets = lib.get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]

    if nms:
        lib.do_nms_obj(dets, num, meta.classes, nms)

    results = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                results.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))

    lib.free_image(im)
    lib.free_detections(dets, num)

    return sorted(results, key=lambda x: -x[1])


def main():
    """
    Main function to process images listed in the validation files and save YOLO detection results.

    This function performs the following steps:
    1. Initializes the YOLO network and metadata.
    2. Reads validation image paths from 'val_from_test.txt' and 'val_from_train.txt'.
    3. Processes each image in the validation set to perform object detection.
    4. Saves the detection results in the specified directories.

    The function uses the YOLO model to detect objects in the images and saves the detection results
    as binary images where detected regions are highlighted.
    """

    # Initialize YOLO network and metadata
    net = lib.load_network(b"~/cfg/yolov3-fishclef.cfg", b"~/fishclef.weights", 0)
    meta = lib.get_metadata(b"~/cfg/fishclef.data")

    # Directories to save YOLO detection results
    save_test_part = "~/Test_dataset/yolo_test_part"
    save_train_part = "~/Test_dataset/yolo_train_part"

    # Read validation image paths
    with open("~/val_from_test.txt") as val_from_test, open(
        "~/val_from_train.txt"
    ) as val_from_train:
        val_test = [line.rstrip() for line in val_from_test]
        val_train = [line.rstrip() for line in val_from_train]

    # Image dimensions
    img_height, img_width = 640, 640
    test_count = 0
    detected_count = 0

    # Process each image in the validation set
    for img_name in val_test:
        test_count += 1
        print(f"Processing {test_count}/{len(val_test)}: {img_name}")
        video_file = os.path.basename(os.path.dirname(img_name))
        img_file = os.path.basename(img_name)

        # Create directory to save results if it doesn't exist
        save_path = join(save_test_part, video_file)
        os.makedirs(save_path, exist_ok=True)

        # Perform detection
        detections = detect(net, meta, img_name)
        detected_blob_img = np.zeros((img_height, img_width), dtype=np.uint8)

        if detections:
            detected_count += 1
            print(f"Detected in frame {detected_count}/{test_count}")
            for fish_info in detections:
                x, y, w, h = map(int, fish_info[2])
                xmin, ymin = max(0, x - w // 2), max(0, y - h // 2)
                xmax, ymax = min(img_width, x + w // 2), min(img_height, y + h // 2)

                # Only consider detections with area less than 25600
                if w * h < 25600:
                    detected_blob_img[ymin:ymax, xmin:xmax] = int(fish_info[1] * 255)

        # Save the detection result image
        cv2.imwrite(join(save_path, img_file), detected_blob_img)


if __name__ == "__main__":
    """
    Entry point of the script.

    This block checks if the script is being run directly (not imported as a module).
    If so, it calls the main() function to start processing the images.
    """
    main()
