"""
A minimalistic interface to run OCR on a given set of images stored locally. See the Predict Notebook for a simple walkthrough.
Note: As of now the line_dilation has to be parameterized until this part of the pipeline has been more generalized. See examples in the Notebook for choosing parameters.

Run e.g. python predict.py --input_dir "Data/W30125" --mode "cuda"
"""

import os
import cv2
import sys
import pyewts
import argparse
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from Utils import create_dir, get_file_name, build_xml_document, back_rotate_lines
from Modules import PatchedLineDetection, OCRInference


line_model_config = "Models/LineModels/line_model_config.json"
ocr_model_config = "Models/OCRModels/LhasaKanjur/ocr_model_config.json"


def run_ocr(image_path: str, out_path: str, save_xml: bool = False, save_preview: bool = False, export_lines: bool = False):
    image_name = get_file_name(image_path)
    image = cv2.imread(image_path)
    
    prediction, rotated_image, line_images, sorted_contours, bbox, peaks, angle = line_inference.predict(image, 0)

    if len(line_images) > 0:
        predicted_text, _ = ocr_inference.run(line_images)
        out_text = f"{out_path}/{image_name}.txt"
        
        with open(out_text, "w", encoding="utf-8") as f:
            for line in predicted_text:
                line = converter.toUnicode(line)
                f.write(f"{line}\n")

        if save_xml:
            rotated_line_contours = back_rotate_lines(
                        image, sorted_contours, angle
                    )
            xml_file = build_xml_document(image, image_name, bbox, rotated_line_contours, out_text)
            
            # TODO: finalize back-rotation for the line contours to avoid saving the rotated image
            #img_out = f"{out_path}/xml/{image_name}.jpg"
            #cv2.imwrite(img_out, rotated_image)
            
            with open(f"{out_path}/page/{image_name}.xml", "w", encoding="utf-8") as f:
                f.write(xml_file)

        if export_lines:
            for idx, line in enumerate(line_images):
                img_out = f"{out_path}/{image_name}_{idx}.jpg"
                cv2.imwrite(img_out, line)

    if save_preview:
        prediction = cv2.cvtColor(prediction, cv2.COLOR_GRAY2BGR)
        cv2.addWeighted(prediction, 0.4, image, 1 - 0.4, 0, image)
        out_prediction = f"{out_path}/{image_name}_prediction.jpg"
        cv2.imwrite(out_prediction, image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", type=str, required=True)
    parser.add_argument("-ext", "--file_ext", type=str, required=False, default="jpg")
    parser.add_argument("-b", "--binarize", choices=["yes", "no"], required=False, default="yes")
    parser.add_argument("-m", "--mode", choices=["cpu", "cuda"], required=False, default="cpu")
    parser.add_argument("-xml", "--xml_output", choices=["yes", "no"], required=False, default="no")
    parser.add_argument("--export_lines", choices=["yes", "no"], required=False, default="no")
    parser.add_argument("--save_preview", choices=["yes", "no"], required=False, default="no")

    args = parser.parse_args()

    input_dir = args.input_dir
    file_ext = args.file_ext
    binarize = args.binarize
    mode = args.mode
    save_xml = True if args.xml_output == "yes" else False
    export_lines = True if args.export_lines == "yes" else False
    save_preview = True if args.save_preview == "yes" else False

    if not os.path.isdir(input_dir):
        sys.exit(f"'{input_dir}' is not a valid directory, cancelling training.")

    input_images = natsorted(glob(f"{input_dir}/*.{file_ext}"))

    out_path = os.path.join(input_dir, "predictions")
    create_dir(out_path)

    if save_xml:
        xml_dir = os.path.join(out_path, "page")
        create_dir(xml_dir)

    if export_lines:
        line_dir = os.path.join(out_path, "lines")
        create_dir(line_dir)

    if not len(input_images) > 0:
        sys.exit(f"{input_dir} does not contain any images.")

    print("starting inference....")
    line_inference = PatchedLineDetection(
        config_file=line_model_config, binarize_output=binarize, mode=mode
    )
    ocr_inference = OCRInference(config_file=ocr_model_config, mode=mode)
    converter = pyewts.pyewts()

    for _, image_path in tqdm(enumerate(input_images), total=len(input_images)):
        run_ocr(image_path, out_path, save_xml=save_xml, save_preview=save_preview, export_lines=export_lines)
    
    print("Done with OCR!")
