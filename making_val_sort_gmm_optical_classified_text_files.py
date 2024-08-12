import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import imageio.v3 as iio

# Constants
VID_DIR = "~/Training_dataset/Videos/"
XML_DIR = "~/Training_dataset/Ground Truth XML/"
SAVE_IMG_DIR = "/home/ahsanjalal/Fishclef/Datasets/Training_dataset/img_pool_retrain1/"
SAVE_LAB_DIR = SAVE_IMG_DIR
SPECIE_LIST = [
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
OTHER_CLASS = "others"
OTHER_LABEL = 15


def process_video(video_path, xml_path, img_counter):
    cap = cv2.VideoCapture(video_path)
    image_vid = []
    success, image = cap.read()
    while success:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_vid.append(image)
        success, image = cap.read()
    cap.release()

    tree = ET.parse(xml_path)
    root = tree.getroot()
    vid_name_short = os.path.basename(video_path).split(".")[0][-15:]

    other_fish_count = 0
    for child in root:
        frame_id = int(child.attrib["id"])
        if frame_id < len(image_vid):
            process_frame(child, frame_id, image_vid, vid_name_short, img_counter)
            img_counter += 1

    return img_counter, other_fish_count


def process_frame(child, frame_id, image_vid, vid_name_short, img_counter):
    for g_child in child:
        fish_specie = g_child.attrib["fish_species"].lower()
        if fish_specie == "chaetodon lununatus":
            fish_specie = "chaetodon lunulatus"

        x, y, w, h = map(
            int,
            [
                g_child.attrib["x"],
                g_child.attrib["y"],
                g_child.attrib["w"],
                g_child.attrib["h"],
            ],
        )
        x, y = max(x, 0), max(y, 0)
        frame = image_vid[frame_id]
        frame = cv2.resize(frame, [640, 640])
        height, width, _ = frame.shape

        mid_x, mid_y = (x + x + w) / (2 * width), (y + y + h) / (2 * height)
        box_width, box_height = w / width, h / height
        filename = f"{vid_name_short}_image_{frame_id}"
        iio.imwrite(f"{SAVE_IMG_DIR}{filename}.jpg", frame)

        fish_lab = (
            SPECIE_LIST.index(fish_specie)
            if fish_specie in SPECIE_LIST
            else OTHER_LABEL
        )
        item = f"{fish_lab} {mid_x} {mid_y} {box_width} {box_height}"
        with open(f"{SAVE_LAB_DIR}{filename}.txt", "a") as a:
            a.write(item + "\n")


def main():
    os.chdir(VID_DIR)
    sub_list = np.array(os.listdir(VID_DIR))
    img_counter = 0

    for vid_count, video in enumerate(sub_list):
        print(f"video number: {vid_count} is in progress")
        video_path = os.path.join(VID_DIR, video)
        xml_path = os.path.join(XML_DIR, os.path.splitext(video)[0] + ".xml")
        img_counter, other_fish_count = process_video(video_path, xml_path, img_counter)

    print("total count for other fish is:", other_fish_count)


if __name__ == "__main__":
    main()
