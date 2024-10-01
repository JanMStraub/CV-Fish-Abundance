import cv2
import numpy as np
import xml.etree.ElementTree as ET


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
    species_list,
    unknown_label,
    image_width,
    image_height,
):
    """
    Generates YOLO format annotations for bounding boxes and saves them to files.

    Args:
        name (str): Name prefix for saved annotation files.
        annotation_file_path (Path): Path where annotation files will be saved.
        bboxes (list): List of bounding boxes for the frame.
        species_list (list): List of species from SPECIES_LIST.
        unknown_label (int): Label for unknown fish species.
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
                species_list.index(fish_species)
                if fish_species in species_list
                else unknown_label
            )
            annotations.append(
                f"{species_index} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            )

        frame_annotation_file = annotation_file_path / f"{name}_{frame_id:04d}.txt"
        with open(frame_annotation_file, "w") as file:
            file.write("\n".join(annotations))


def extract_ground_truth(video_path, species_key, gt_dir):
    """
    Extracts ground truth annotations from the corresponding XML file.

    Args:
        video_path (Path): Path to the video file.
        species_key (str): Key for accessing species name in bbox dictionary (default is 'fish_species').
        gt_dir (Path): Path to the ground truth file.

    Returns:
        list: List of ground truth bounding boxes extracted from XML.
    """
    file_name_without_ext = video_path.stem
    gt_xml_path = gt_dir / f"{file_name_without_ext}.xml"

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
