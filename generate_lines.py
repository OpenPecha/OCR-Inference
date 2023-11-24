"""
A minimalistic interface for generating line images from a given input directory. 

"""

import os
import cv2
import sys
import argparse
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from Utils import create_dir, get_file_name
from Modules import PatchedLineDetection


line_model_config = "Models/LineModels/line_model_config.json"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--file_ext", type=str, required=False, default="jpg")
    parser.add_argument(
        "--mode", choices=["cpu", "cuda"], required=False, default="cpu"
    )
    parser.add_argument(
        "--binarize", choices=["yes", "no"], required=False, default="yes"
    )
    parser.add_argument(
        "--save_preview", choices=["yes", "no"], required=False, default="no"
    )

    args = parser.parse_args()

    input_dir = args.input_dir
    file_ext = args.file_ext
    mode = args.mode
    binarize = True if args.binarize == "yes" else False
    save_preview = True if args.save_preview == "yes" else False

    if not os.path.isdir(input_dir):
        sys.exit(f"'{input_dir}' is not a valid directory, cancelling training.")

    input_images = natsorted(glob(f"{input_dir}/*.{file_ext}"))

    out_path = os.path.join(input_dir, "lines")
    create_dir(out_path)

    if save_preview:
        create_dir(os.path.join(out_path, "previews"))

    if not len(input_images) > 0:
        sys.exit(f"'{input_dir}' does not contain any images.")

    print("starting inference....")
    line_inference = PatchedLineDetection(
        config_file=line_model_config, binarize_output=binarize, mode=mode
    )

    for _, image_path in tqdm(enumerate(input_images), total=len(input_images)):
        image_name = get_file_name(image_path)
        image = cv2.imread(image_path)
        prediction, line_images, sorted_contours, bbox, peaks = line_inference.predict(
            image, 0
        )

        for idx, line in enumerate(line_images):
            img_out = f"{out_path}/{image_name}_{idx}.jpg"
            cv2.imwrite(img_out, line)

        if save_preview:
            prediction = cv2.cvtColor(prediction, cv2.COLOR_GRAY2BGR)
            cv2.addWeighted(prediction, 0.4, image, 1 - 0.4, 0, image)
            out_prediction = f"{out_path}/previews/{image_name}_preview.jpg"
            cv2.imwrite(out_prediction, image)
