import os
import glob
import random
import cv2
import numpy as np
from ctypes import *
from os.path import join

# Constants and Globals
bkg_count = 0
total_gmm_count = 0
total_gt_count = 0
TP = 0
FP = 0
gmm_count = 0
num = np.zeros(16)  # For UWA dataset

# Directory Paths
gmm_main_dir = "~/gmm_optical_combined"
gt_main_dir = "~/annotated_frames"
save_main_dir = "~/gmm_optical_combined_output"
saving_dir = "~/test_frames"
gt_fol = os.listdir(gt_main_dir)

# Species List
specie_list = [
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
    "Background",
]

# Load YOLO model
lib = CDLL("~/libdarknet.so", RTLD_GLOBAL)
net = lib.load_network("~/resnet50.cfg", "~/resnet50_146.weights", 0)
meta = lib.get_metadata("~/fish_classification.data")


# Function Definitions
def sample(probs):
    s = sum(probs)
    probs = [a / s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r -= probs[i]
        if r <= 0:
            return i
    return len(probs) - 1


def c_array(ctype, values):
    arr = (ctype * len(values))()
    arr[:] = values
    return arr


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


def classify(net, meta, im):
    out = lib.network_predict_image(net, im)
    res = [(meta.names[i], out[i]) for i in range(meta.classes)]
    return sorted(res, key=lambda x: -x[1])


def detect(net, meta, image, thresh=0.5, hier_thresh=0.5, nms=0.45):
    im = lib.load_image_color(image.encode("utf-8"), 0, 0)
    pnum = pointer(c_int(0))
    lib.network_predict_image(net, im)
    dets = lib.get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if nms:
        lib.do_nms_obj(dets, num, meta.classes, nms)

    res = [
        (
            meta.names[i],
            dets[j].prob[i],
            (dets[j].bbox.x, dets[j].bbox.y, dets[j].bbox.w, dets[j].bbox.h),
        )
        for j in range(num)
        for i in range(meta.classes)
        if dets[j].prob[i] > 0
    ]
    lib.free_image(im)
    lib.free_detections(dets, num)
    return sorted(res, key=lambda x: -x[1])


# Main Processing Loop
for video_fol in gt_fol:
    print(f"Processing video {video_fol}")
    vid_fol_path = join(gt_main_dir, video_fol)
    os.chdir(vid_fol_path)

    gt_text_files = glob.glob("*.txt")
    gt_height, gt_width = [640, 640]
    gmm_height, gmm_width = [640, 640]

    for gt_file in gt_text_files:
        img_gt = cv2.imread(gt_file.replace(".txt", ".png"))
        with open(gt_file) as f:
            gt_text = f.readlines()
        gt_count = len(gt_text)
        total_gt_count += gt_count

        gmm_file_path = join(gmm_main_dir, video_fol, gt_file)
        if os.path.isfile(gmm_file_path):
            with open(gmm_file_path) as f:
                text_gmm = f.readlines()
            img_gmm = cv2.imread(gmm_file_path.replace(".txt", ".png"))

            for line_gmm in text_gmm:
                gmm_count += 1
                total_gmm_count += 1
                x_gmm, y_gmm, w_gmm, h_gmm = parse_coords(
                    line_gmm, gmm_width, gmm_height
                )
                coords_gmm = adjust_coords(
                    x_gmm, y_gmm, w_gmm, h_gmm, gmm_width, gmm_height
                )

                match_flag = 0
                for count_gt_line, line_gt in enumerate(gt_text):
                    x_gt, y_gt, w_gt, h_gt = parse_coords(line_gt, gt_width, gt_height)
                    coords_gt = adjust_coords(
                        x_gt, y_gt, w_gt, h_gt, gt_width, gt_height
                    )
                    if is_match(coords_gmm, coords_gt):
                        match_flag += 1
                        TP += 1
                        save_detection(
                            img_gt,
                            coords_gt,
                            gt_file,
                            video_fol,
                            num,
                            saving_dir,
                            specie_list,
                        )
                        del gt_text[count_gt_line]
                        break

                if not match_flag:
                    FP += handle_false_positive(
                        img_gt, coords_gmm, save_main_dir, net, meta
                    )

        else:
            FP += gt_count

# Calculate and print metrics
FN = abs(total_gt_count - TP)
PR = TP / (TP + FP)
RE = TP / (TP + FN)
F_SCORE = 2 * PR * RE / (PR + RE)

print(f"True Positives: {TP}")
print(f"False Positives: {FP}")
print(f"False Negatives: {FN}")
print(f"Precision: {PR}")
print(f"Recall: {RE}")
print(f"F-score: {F_SCORE}")
print(f"Background FP removed: {bkg_count}")


# Helper Functions
def parse_coords(line, width, height):
    coords = line.strip().split(" ")
    x = int(round(float(coords[1]) * width))
    y = int(round(float(coords[2]) * height))
    w = int(round(float(coords[3]) * width))
    h = int(round(float(coords[4]) * height))
    return x, y, w, h


def adjust_coords(x, y, w, h, img_width, img_height):
    xmin = max(0, x - w // 2)
    ymin = max(0, y - h // 2)
    xmax = min(img_width, x + w // 2)
    ymax = min(img_height, y + h // 2)
    return xmin, ymin, xmax, ymax


def is_match(coords_gmm, coords_gt, threshold=0.5):
    xa, ya = max(coords_gmm[0], coords_gt[0]), max(coords_gmm[1], coords_gt[1])
    xb, yb = min(coords_gmm[2], coords_gt[2]), min(coords_gmm[3], coords_gt[3])
    if xb > xa and yb > ya:
        inter_area = (xb - xa + 1) * (yb - ya + 1)
        gt_area = (coords_gt[2] - coords_gt[0] + 1) * (coords_gt[3] - coords_gt[1] + 1)
        pred_area = (coords_gmm[2] - coords_gmm[0] + 1) * (
            coords_gmm[3] - coords_gmm[1] + 1
        )
        min_area = min(gt_area, pred_area)
        return float(inter_area) / min_area >= threshold
    return False


def save_detection(img, coords, gt_file, video_fol, num, saving_dir, specie_list):
    fish_label = int(gt_file.split(" ")[0])
    img_patch = cv2.resize(
        img[coords[1] : coords[3], coords[0] : coords[2]].astype("float32"),
        dsize=(50, 50),
    )
    fish_name = specie_list[fish_label].split(" ")[1]
    save_path = join(saving_dir, video_fol)
    os.makedirs(save_path, exist_ok=True)
    cv2.imwrite(join(save_path, f"{num[fish_label]}_{fish_name}.png"), img_patch)
    num[fish_label] += 1


def handle_false_positive(img, coords, save_main_dir, net, meta):
    global bkg_count
    img_patch = cv2.resize(
        img[coords[1] : coords[3], coords[0] : coords[2]].astype("float32"),
        dsize=(50, 50),
    )
    os.makedirs(save_main_dir, exist_ok=True)
    test_img_path = join(save_main_dir, "test_image.png")
    cv2.imwrite(test_img_path, img_patch)
    im = lib.load_image_color(test_img_path.encode("utf-8"), 0, 0)
    r = classify(net, meta, im)
    if r[0][0] == "background" or r[0][1] < 0.9:
        bkg_count += 1
        return 0
    return 1
