
"""
A minimalistic interface for mapping existing e-text to predicted lines.
The interface assumes  len(labels) == len(images), so there some data preprocessing and optional cleanup will be necessary.
This mapping is pretty much naive and mismatches could occur for several reasons since there is only a simple comparision for the 
number of lines in the e-text file and the number of lines returend by the line detection.
"""



import os
import cv2
import json
import pyewts
import argparse
from tqdm import tqdm
from glob import glob
from natsort import natsorted
from Utils import get_lines_from_page, create_dir

from Modules import PatchedLineDetection, OCRInference
from evaluate import load



def map_data(line_inference, ocr_inference, volume_dir: str, ext: str = "jpg"):
    print(volume_dir)
    #label_path = os.path.join(volume_dir, "transcriptions")
    label_path = f"{volume_dir}/transcriptions"
    images = natsorted(glob(f"{volume_dir}/*.{ext}"))
    labels = natsorted(glob(f"{label_path}/*.txt"))

    print(f"Images: {len(images)}, Labels: {len(labels)}")

    if len(images) > 0 and len(labels) > 0:
        assert(len(images) == len(labels))

    try:
        assert(len(images) > 0 and len(labels) > 0)
        assert(len(images) == len(labels))
    except AssertionError:
        print("Images and label count not matching!")
        print(f"Found {len(images)} images and {len(labels)} labels.")


    ### run mapping function on entire dataset
    line_dataset_out = os.path.join(volume_dir, "Dataset")
    dataset_imgs = os.path.join(line_dataset_out, "lines")
    dataset_lbls = os.path.join(line_dataset_out, "transcriptions")
    line_prevs = os.path.join(line_dataset_out, "previews")

    create_dir(dataset_imgs)
    create_dir(dataset_lbls)
    create_dir(line_prevs)

    line_mismatches = []
    ds_dict = []
    alpha = 0.4


    for page_idx, (image_path, label_path) in tqdm(enumerate(zip(images, labels)), total=len(images)):
        filename = os.path.basename(image_path).split(".")[0]
        image = cv2.imread(image_path)
        wylie_labels = get_lines_from_page(label_path)
        wylie_labels = [converter.toWylie(x) for x in wylie_labels]
        wylie_labels = [x.replace("_", " ")   for x in wylie_labels]

        prediction, rotated_image, line_images, sorted_contours, bbox, peaks, angle  = line_inference.predict(image, 0, class_threshold=0.5)
        prediction = cv2.cvtColor(prediction, cv2.COLOR_GRAY2RGB)
        prev_img = image.copy()
        cv2.addWeighted(prediction, alpha, prev_img, 1 - alpha, 0, prev_img)

        line_preview_out = f"{line_prevs}/{filename}.jpg"
        cv2.imwrite(line_preview_out, prev_img)

        if (len(wylie_labels) == len(line_images)):
            predicted_text, _ = ocr_inference.run(line_images)

            for line_idx, (label, pred_label, line_image) in enumerate(zip(wylie_labels, predicted_text, line_images)):
                cer_score = cer_scorer.compute(predictions=[pred_label], references=[label])
                cer_score = round(cer_score, 2)

                record = {
                    "index": f"{filename}_{line_idx}",
                    "cer": cer_score,
                    "gt": label,
                    "pred": pred_label,
                    }
                
                ds_dict.append(record)

                #ds_dict[f"{idx}_{line_idx}"] = (cer_score, line_idx, label)

                label_out = f"{dataset_lbls}/{filename}_{line_idx}.txt"
                img_out = f"{dataset_imgs}/{filename}_{line_idx}.jpg"

                cv2.imwrite(img_out, line_image)

                with open(label_out, "w", encoding="utf-8") as f:
                    f.write(label)

        else:
            line_mismatches.append(image_path)


    mismatch_file_out = f"{line_dataset_out}/line_mismatches.txt"

    with open(mismatch_file_out, "w", encoding="utf-8") as f:
        for entry in line_mismatches:
            f.write(f"{entry}\n")


    dataset_dict_out = f"{line_dataset_out}/score_dict.json"

    #with open(dataset_dict_out, "w", encoding="utf-8") as f:
    #    for key, entry in ds_dict.items():
    #       f.write(f"{key}: {entry[0]}, {entry[1], [entry[2]]}\n")

    with open(dataset_dict_out, "w", encoding='UTF-8') as f:
        json.dump(ds_dict, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", type=str, required=True)
    parser.add_argument("-v", "--volumes", type=str, required=False, default="")
    parser.add_argument("-ext", "--file_ext", type=str, required=False, default="jpg")
    parser.add_argument("-m", "--mode", choices=["cpu", "cuda"], required=False, default="cpu")
    parser.add_argument("-b", "--binarize", choices=["yes", "no"], required=False, default="yes")

    args = parser.parse_args()
    input_dir = args.input_dir
    volumes = args.volumes
    file_ext = args.file_ext
    mode = args.mode
    binarize = True if args.binarize == "yes" else False


    converter = pyewts.pyewts()
    cer_scorer = load("cer")
    line_model_config = "Models/LineModels/line_model_config.json"
    ocr_model_config = "Models/OCRModels/LhasaKanjur/ocr_model_config.json"


    line_inference = PatchedLineDetection(config_file=line_model_config, binarize_output=binarize, mode="cuda")
    ocr_inference = OCRInference(config_file=ocr_model_config, mode="cuda")

    if volumes != "":
        volumes = volumes.split(",")
        volumes = [x.strip() for x in volumes]
        print(volumes)
        #volumes = ["I4090", "I4089", "I4088", "I4087", "I4086", "I4085", "I4084", "I4083", "I4082", "I4081", "I4080", "I4079", "I4078", "I4077", "I4076", "I4075", "I4074", "I4073", "I4072", "I4071", "I4070", "I4069", "I4068", "I4067"]
    
        for volume in volumes:
            #volume_dir = os.path.join(input_dir, volume)
            volume_dir = f"{input_dir}/{volume}"
    
            if os.path.isdir(volume_dir):
                map_data(line_inference, ocr_inference, volume_dir, file_ext)
            else:
                print(f"{volume_dir} is not a valid directory, skipping!")
                