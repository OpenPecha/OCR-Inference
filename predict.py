"""
A minimalistic interface to run OCR on a given set of images stored locally. See the Predict Notebook for a simple walkthrough.
Note: As of now the line_dilation has to be parameterized until this part of the pipeline has been more generalized. See examples in the Notebook for choosing parameters.

Run e.g. python predict.py --input_dir "Data/W30125" --line_dilate 10 --line_iteration 10

"""
import os
import cv2
import sys
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt

from glob import glob
from tqdm import tqdm

from natsort import natsorted

from Modules import LineDetection, OCRInference
from Utils import create_dir, get_file_name

line_model_config = "Models/LineModels/line_model_config.json"
ocr_model_config = "Models/OCRModels/LhasaKanjur/model_config.json"


def run_ocr(image_path: str, out_path: str, save_preview: bool = False):
    image_name = get_file_name(image_path)
    image = cv2.imread(image_path)
    prediction, line_images, sorted_contours, peaks = line_inference.predict(image, 0)
    predicted_text = ocr_inference.run(line_images)

    out_text = f"{out_path}/{image_name}.txt"

    with open(out_text, "w", encoding="utf-8") as f:
        for line in predicted_text:
            f.write(f"{line}\n")

    if save_preview:
        prediction = cv2.cvtColor(prediction, cv2.COLOR_GRAY2BGR)
        cv2.addWeighted(prediction, 0.4, image, 1 - 0.4, 0, image)
        out_prediction = f"{out_path}/{image_name}_prediction.jpg"
        cv2.imwrite(out_prediction, image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--line_dilate", type=int, required=False, default=32)
    parser.add_argument("--line_iteration", type=int, required=False, default=30)

    args = parser.parse_args()

    input_dir = args.input_dir
    line_dilate = args.line_dilate
    line_iteration = args.line_iteration

    if not os.path.isdir(input_dir):
        sys.exit(f"'{input_dir}' is not a valid directory, cancelling training.")

    input_images = natsorted(glob(f"{input_dir}/*.jpg"))

    out_path = os.path.join(input_dir, "predictions")
    create_dir(out_path)

    if not len(input_images) > 0:
        sys.exit(f"'{input_dir}' does not contain any images.")

    ### use this for training a network from scratch
    logging.info("starting inference....")
    line_inference = LineDetection(
        config_file=line_model_config,
        dilate_kernel=line_dilate,
        dilate_iterations=line_iteration,
        binarize_output=False,
    )
    ocr_inference = OCRInference(config_file=ocr_model_config)

    for _, image_path in tqdm(enumerate(input_images), total=len(input_images)):
        run_ocr(image_path, out_path, save_preview=True)
        logging.info("Done with OCR!")
