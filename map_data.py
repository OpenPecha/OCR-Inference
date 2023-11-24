import os
import cv2
import json
import pyewts
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob
from natsort import natsorted
from Utils import get_lines_from_page, create_dir, binarize_line

from Modules import PatchedLineDetection, OCRInference
from evaluate import load


converter = pyewts.pyewts()
cer_scorer = load("cer")


line_model_config = "Models\LineModels\line_model_config.json"
line_inference = PatchedLineDetection(
        config_file=line_model_config, binarize_output=False, mode="cuda"
    )

ocr_model_config = "Models/OCRModels/LhasaKanjur/ocr_model_config.json"
ocr_inference = OCRInference(config_file=ocr_model_config, mode="cuda")



data_root = "Data/W23703/I1320"
#image_path = os.path.join(data_root, "pages")
image_path = data_root
label_path = os.path.join(data_root, "transcriptions")

images = natsorted(glob(f"{image_path}/*.jpg"))
labels = natsorted(glob(f"{label_path}/*.txt"))

print(f"Images: {len(images)}, Labels: {len(labels)}")


### run mapping function on entire dataset
line_dataset_out = os.path.join(data_root, "DatasetNov23")
dataset_imgs = os.path.join(line_dataset_out, "lines")
dataset_lbls = os.path.join(line_dataset_out, "transcriptions")
line_prevs = os.path.join(line_dataset_out, "previews")

create_dir(dataset_imgs)
create_dir(dataset_lbls)


line_mismatches = []
ds_dict = []
alpha = 0.4


def map_data():
    for page_idx, (image_path, label_path) in tqdm(enumerate(zip(images, labels)), total=len(images)):
        filename = os.path.basename(image_path).split(".")[0]
        image = cv2.imread(image_path)
        wylie_labels = get_lines_from_page(label_path)
        wylie_labels = [converter.toWylie(x) for x in wylie_labels]
        wylie_labels = [x.replace("_", " ")   for x in wylie_labels]

        prediction, line_images, sorted_contours, bbox, peaks = line_inference.predict(image, 0, class_threshold=0.5, binarize=False)
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
    map_data()

