import os
import glob
import numpy as np
import cv2
import random
from ctypes import *
from collections import Counter
from imutils.face_utils import FaceAligner, rect_to_bb
from natsort import natsorted
from shutil import copytree


# Darknet configurations and utilities
def sample(probs):
    s = sum(probs)
    probs = [a / s for a in probs]
    r = random.uniform(0, 1)
    for i, prob in enumerate(probs):
        r -= prob
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


# Load darknet library
lib = CDLL("~/libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [
    c_void_p,
    c_int,
    c_int,
    c_float,
    c_float,
    POINTER(c_int),
    c_int,
    POINTER(c_int),
]
get_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)


def classify(net, meta, im):
    out = predict_image(net, im)
    res = [(meta.names[i], out[i]) for i in range(meta.classes)]
    return sorted(res, key=lambda x: -x[1])


def detect(net, meta, image, thresh=0.5, hier_thresh=0.5, nms=0.45):
    im = load_image(image, 0, 0)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if nms:
        do_nms_obj(dets, num, meta.classes, nms)

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
    free_image(im)
    free_detections(dets, num)
    return sorted(res, key=lambda x: -x[1])


# Load network and metadata
net = load_net("~/resnet50.cfg", "~/resnet50_146.weights", 0)
meta = load_meta("~/fish_classification.data")

# Directories and initialization
total_gmm_count, total_gt_count, TP, FP, bkg_count = 0, 0, 0, 0, 0
gmm_main_dir = "~/gmm_optical_combined"
gt_main_dir = "~/annotated_frames"
save_main_dir = "~/gmm_optical_combined_output"
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
num = np.zeros(len(specie_list))  # Initialize counter for each species
gt_fol = os.listdir(gt_main_dir)

for vid_counter, video_fol in enumerate(gt_fol):
    print(f"Processing video {vid_counter}: {video_fol}")
    vid_fol_path = os.path.join(gt_main_dir, video_fol)
    os.chdir(vid_fol_path)

    gt_text_files = glob.glob("*.txt")
    gt_height, gt_width = 640, 640
    gmm_height, gmm_width = 640, 640

    for gt_files in gt_text_files:
        img_gt = cv2.imread(gt_files.replace(".txt", ".png"))
        with open(gt_files) as a:
            gt_text = a.readlines()

        gt_count = len(gt_text)
        total_gt_count += gt_count

        gmm_file_path = os.path.join(gmm_main_dir, video_fol, gt_files)
        if os.path.isfile(gmm_file_path):
            with open(gmm_file_path) as gmm_text_file:
                text_gmm = gmm_text_file.readlines()

            img_gmm = cv2.imread(gmm_file_path.replace(".txt", ".png"))

            for line_gmm in text_gmm:
                gmm_count += 1
                total_gmm_count += 1

                coords_gmm = list(map(float, line_gmm.strip().split()))
                w_gmm, h_gmm = round(coords_gmm[3] * gmm_width), round(
                    coords_gmm[4] * gmm_height
                )
                x_gmm, y_gmm = round(coords_gmm[1] * gmm_width), round(
                    coords_gmm[2] * gmm_height
                )
                xmin_gmm, ymin_gmm = max(0, x_gmm - w_gmm // 2), max(
                    0, y_gmm - h_gmm // 2
                )
                xmax_gmm, ymax_gmm = min(gmm_width, x_gmm + w_gmm // 2), min(
                    gmm_height, y_gmm + h_gmm // 2
                )

                match_flag = False
                for line_gt in gt_text[
                    :
                ]:  # Copy the list to avoid issues with deletion during iteration
                    coords_gt = list(map(float, line_gt.strip().split()))
                    fish_label = int(coords_gt[0])

                    w_gt, h_gt = round(coords_gt[3] * gt_width), round(
                        coords_gt[4] * gt_height
                    )
                    x_gt, y_gt = round(coords_gt[1] * gt_width), round(
                        coords_gt[2] * gt_height
                    )
                    xmin_gt, ymin_gt = max(0, x_gt - w_gt // 2), max(
                        0, y_gt - h_gt // 2
                    )
                    xmax_gt, ymax_gt = min(gt_width, x_gt + w_gt // 2), min(
                        gt_height, y_gt + h_gt // 2
                    )

                    xa, ya, xb, yb = (
                        max(xmin_gmm, xmin_gt),
                        max(ymin_gmm, ymin_gt),
                        min(xmax_gmm, xmax_gt),
                        min(ymax_gmm, ymax_gt),
                    )
                    if xb > xa and yb > ya:
                        match_flag = True
                        gt_text.remove(line_gt)
                        num[fish_label] += 1
                        TP += 1

                        os.makedirs(
                            os.path.join(save_main_dir, specie_list[fish_label]),
                            exist_ok=True,
                        )
                        save_image_path = os.path.join(
                            save_main_dir,
                            specie_list[fish_label],
                            f"{num[fish_label]:04d}.png",
                        )
                        cropped_image = img_gt[ymin_gt:ymax_gt, xmin_gt:xmax_gt]
                        cv2.imwrite(save_image_path, cropped_image)
                        break

                if not match_flag:
                    FP += 1

    for remaining_line in gt_text:
        bkg_count += 1
        coords_gt = list(map(float, remaining_line.strip().split()))
        fish_label = int(coords_gt[0])

        w_gt, h_gt = round(coords_gt[3] * gt_width), round(coords_gt[4] * gt_height)
        x_gt, y_gt = round(coords_gt[1] * gt_width), round(coords_gt[2] * gt_height)
        xmin_gt, ymin_gt = max(0, x_gt - w_gt // 2), max(0, y_gt - h_gt // 2)
        xmax_gt, ymax_gt = min(gt_width, x_gt + w_gt // 2), min(
            gt_height, y_gt + h_gt // 2
        )

        os.makedirs(os.path.join(save_main_dir, "Background"), exist_ok=True)
        save_image_path = os.path.join(
            save_main_dir, "Background", f"{bkg_count:04d}.png"
        )
        cropped_image = img_gt[ymin_gt:ymax_gt, xmin_gt:xmax_gt]
        cv2.imwrite(save_image_path, cropped_image)
