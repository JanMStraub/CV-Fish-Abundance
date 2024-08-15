# HOGY Toolbox for Fish Detection and Categorization

This algorithm detects and classifies fish instances in unconstrained environments using a hybrid approach combining GMM, Optical Flow, and a deep CNN based on YOLO. Preference is given to YOLO during hybridization when results from GMM-Optical and YOLO overlap.

## Update 2024

For the lecture 'Computer Vision' at the University of Heidelberg, Germany, this repository is going to be updated.
The datasets that we used in out project are the Fish4Knowledge sets for [fish detection and fish species recognition](http://www.perceivelab.com/datasets).
Other fish related datasets can be found [here](https://globalwetlandsproject.org/computer-vision-resources-fish-classification-datasets/) or [here](https://github.com/Callmewuxin/fish4konwledge/tree/master) or [here](https://public.roboflow.com/object-detection/aquarium).

## Making Frames from Videos

To save GT frames of the LCF-15 dataset, use `making_GT_frames_lcf15.py` on the dataset. For the UWA dataset, the dataset will be provided upon request: [ahsan.jalal@seecs.edu.pk](mailto:ahsan.jalal@seecs.edu.pk), [ahmad.salman@seecs.edu.pk](mailto:ahmad.salman@seecs.edu.pk).

## GMM Output

Run `GMM/GMM_frames_per_video.m` to save GMM frames for all videos along with annotated text files. This script is written in Matlab.

## Optical Flow Output

Run `Optical_flow/optical_flow_frames_per_video.py` to save Optical Flow for the required frames. This script is written in Python.

## YOLO DNN

For YOLO, clone this repository: [https://github.com/AlexeyAB/darknet.git](https://github.com/AlexeyAB/darknet.git) and build it according to the instructions on [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet) (use `libso=1` in the Makefile).

### Steps to Follow:

1. **Make Training Data List**: Follow the instructions in the YOLO repository to create a training data list.
   
2. **Edit YOLO Configuration**: 
   - Edit the `yolov3.cfg` file for the LCF-15 and UWA datasets (15 & 16 classes in LCF-15 and UWA datasets respectively).
   
3. **Create `.names` Files**:
   - Make separate `.names` files for the LCF-15 and UWA datasets. Include all class names as specified in the YOLO instructions.
   
4. **Create `.data` Files**:
   - Make separate `.data` files for each dataset.
   - Copy contents from the `coco.data` file in the `yolo/cfg` directory into each new file.
   - Edit the `classes`, `train`, `names`, and `backup` variables accordingly.

### Evaluation

You need a pre-trained model on the respective datasets. These models will be shared upon request: [ahsan.jalal@seecs.edu.pk](mailto:ahsan.jalal@seecs.edu.pk), [ahmad.salman@seecs.edu.pk](mailto:ahmad.salman@seecs.edu.pk).

Once you have the models and test splits, use `YOLO_DNN/yolo_on_frames.py` to save classification results.

## Combining Outputs

Use `making_gmm_optical_gray_combined_image.py` to combine GMM and Optical Flow outputs into one 2D frame (green channel for GMM and red channel for Optical Flow).

## Classification with ResNet-50

ResNet-50 models trained on the LCF-15 and UWA datasets are required to classify objects detected by the GMM & Optical combined output. Models will be shared upon request: [ahsan.jalal@seecs.edu.pk](mailto:ahsan.jalal@seecs.edu.pk), [ahmad.salman@seecs.edu.pk](mailto:ahmad.salman@seecs.edu.pk).

Once you have the models, use `making_val_sort_gmm_optical_classified_text_files.py` to save classification results on the GMM & Optical combined input.

## F-Score Calculation

Use `val_sort_gmm_optical_vs_yolo_f_score.py` to calculate the F-score for the given dataset using GMM-Optical and YOLO classified outputs. This score will be compared against ground truths (GTs). Preference is given to YOLO output when results overlap with GMM-Optical.