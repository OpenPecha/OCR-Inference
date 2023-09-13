import os
import cv2
import math
import numpy as np
import tensorflow as tf
from typing import Tuple


def get_file_name(x) -> str:
    return os.path.basename(x).split(".")[0]


def create_dir(dir_name: str) -> None:
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def resize_image(
    orig_img: np.array, target_width: int = 2048
) -> Tuple[np.array, float]:
    if orig_img.shape[1] > orig_img.shape[0]:
        resize_factor = round(target_width / orig_img.shape[1], 2)
        target_height = int(orig_img.shape[0] * resize_factor)

        resized_img = cv2.resize(orig_img, (target_width, target_height))

    else:
        target_height = target_width
        resize_factor = round(target_width / orig_img.shape[0], 2)
        target_width = int(orig_img.shape[1] * resize_factor)
        resized_img = cv2.resize(orig_img, (target_width, target_height))

    return resized_img, resize_factor


def pad_image(
    img: np.array, patch_size: int = 64, is_mask=False
) -> Tuple[np.array, Tuple[float, float]]:
    x_pad = (math.ceil(img.shape[1] / patch_size) * patch_size) - img.shape[1]
    y_pad = (math.ceil(img.shape[0] / patch_size) * patch_size) - img.shape[0]

    if is_mask:
        pad_y = np.zeros(shape=(y_pad, img.shape[1], 3), dtype=np.uint8)
        pad_x = np.zeros(shape=(img.shape[0] + y_pad, x_pad, 3), dtype=np.uint8)
    else:
        pad_y = np.ones(shape=(y_pad, img.shape[1], 3), dtype=np.uint8)
        pad_x = np.ones(shape=(img.shape[0] + y_pad, x_pad, 3), dtype=np.uint8)
        pad_y *= 255
        pad_x *= 255

    img = np.vstack((img, pad_y))
    img = np.hstack((img, pad_x))

    return img, (x_pad, y_pad)


def resize_to_height(image, target_height: int):
    width_ratio = target_height / image.shape[0]
    image = cv2.resize(image, (int(image.shape[1] * width_ratio), target_height), interpolation=cv2.INTER_LINEAR)
    return image


def resize_to_width(image, target_width: int):
    width_ratio = target_width / image.shape[1]
    image = cv2.resize(image, (target_width, int(image.shape[0] * width_ratio)), interpolation=cv2.INTER_LINEAR)
    return image


def pad_image2(
    img: np.array, target_width: int, target_height: int, padding: str
) -> np.array:
    width_ratio = target_width / img.shape[1]
    height_ratio = target_height / img.shape[0]

    if width_ratio < height_ratio:  # maybe handle equality separately
        tmp_img = resize_to_width(img, target_width)

        if padding == "white":
            v_stack = np.ones(
                (target_height - tmp_img.shape[0], tmp_img.shape[1]), dtype=np.uint8
            )
        else:
            v_stack = np.zeros(
                (target_height - tmp_img.shape[0], tmp_img.shape[1]), dtype=np.uint8
            )
        v_stack *= 255

        out_img = np.vstack([v_stack, tmp_img])

    elif width_ratio > height_ratio:
        tmp_img = resize_to_height(img, target_height)

        if padding == "white":
            h_stack = np.ones(
                (tmp_img.shape[0], target_width - tmp_img.shape[1]), dtype=np.uint8
            )
        else:
            h_stack = np.zeros(
                (tmp_img.shape[0], target_width - tmp_img.shape[1]), dtype=np.uint8
            )
        h_stack *= 255

        out_img = np.hstack([tmp_img, h_stack])
    else:
        tmp_img = resize_to_width(img, target_width)

        if padding == "white":
            v_stack = np.ones(
                (target_height - tmp_img.shape[0], tmp_img.shape[1]), dtype=np.uint8
            )
        else:
            v_stack = np.zeros(
                (target_height - tmp_img.shape[0], tmp_img.shape[1]), dtype=np.uint8
            )
        v_stack *= 255

        out_img = np.vstack([v_stack, tmp_img])
        print(
            f"Info -> equal ratio: {img.shape}, w_ratio: {width_ratio}, h_ratio: {height_ratio}"
        )

    return cv2.resize(out_img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)


def patch_image(
    img: np.array, patch_size: int = 64, overlap: int = 2, is_mask=False
) -> list:
    """
    A simple slicing function.
    Expects input_image.shape[0] and image.shape[1] % patch_size = 0
    """

    y_steps = img.shape[0] // patch_size
    x_steps = img.shape[1] // patch_size

    patches = []

    for y_step in range(0, y_steps):
        for x_step in range(0, x_steps):
            x_start = x_step * patch_size
            x_end = (x_step * patch_size) + patch_size

            crop_patch = img[
                y_step * patch_size : (y_step * patch_size) + patch_size, x_start:x_end
            ]
            patches.append(crop_patch)

    return patches, y_steps


def unpatch_image(image, pred_patches: list) -> np.array:
    patch_size = pred_patches[0].shape[1]

    x_step = math.ceil(image.shape[1] / patch_size)

    list_chunked = [
        pred_patches[i : i + x_step] for i in range(0, len(pred_patches), x_step)
    ]

    final_out = np.zeros(shape=(1, patch_size * x_step))

    for y_idx in range(0, len(list_chunked)):
        x_stack = list_chunked[y_idx][0]

        for x_idx in range(1, len(list_chunked[y_idx])):
            patch_stack = np.hstack((x_stack, list_chunked[y_idx][x_idx]))
            x_stack = patch_stack

        final_out = np.vstack((final_out, x_stack))

    final_out = final_out[1:, :]
    final_out *= 255

    return final_out


def unpatch_prediction(prediction: np.array, y_splits: int) -> np.array:
    prediction *= 255
    prediction_sliced = np.array_split(prediction, y_splits, axis=0)
    prediction_sliced = [np.concatenate(x, axis=1) for x in prediction_sliced]
    prediction_sliced = np.vstack(np.array(prediction_sliced))

    return prediction_sliced


def rotate_page(
    original_image: np.array,
    line_mask: np.array,
    max_angle: float = 3.0,
    debug_angles: bool = False,
) -> float:
    contours, _ = cv2.findContours(line_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    angles = [cv2.minAreaRect(x)[2] for x in contours]

    # angles = [x for x in angles if abs(x) != 0.0 and x != 90.0]
    low_angles = [x for x in angles if abs(x) != 0.0 and x < max_angle]
    high_angles = [x for x in angles if abs(x) != 90.0 and x > (90 - max_angle)]

    if debug_angles:
        print(angles)

    if len(low_angles) > len(high_angles) and len(low_angles) > 0:
        mean_angle = np.mean(low_angles)

    # check for clockwise rotation
    elif len(high_angles) > 0:
        mean_angle = -(90 - np.mean(high_angles))

    else:
        mean_angle = 0

    rows, cols = original_image.shape[:2]
    rot_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), mean_angle, 1)
    rotated_img = cv2.warpAffine(
        original_image, rot_matrix, (cols, rows), borderValue=(0, 0, 0)
    )

    rotated_prediction = cv2.warpAffine(
        line_mask, rot_matrix, (cols, rows), borderValue=(0, 0, 0)
    )

    return rotated_img, rotated_prediction, mean_angle


def optimize_countour(cnt, e=0.001):
    epsilon = e * cv2.arcLength(cnt, True)
    return cv2.approxPolyDP(cnt, epsilon, True)


def prepare_ocr_image(
    line_image: np.array,
    target_width: int = 2000,
    target_height: int = 80,
    padding: str = "black",
):
    if len(line_image.shape) > 2:
        line_image = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY)
    line_image = pad_image2(line_image, target_width, target_height, padding=padding)
    line_image = line_image.astype(np.float32)
    line_image /= 255.0
    line_image = np.transpose(line_image, axes=[1, 0])

    return line_image

"""
Note: There is some yet unclear behaviour that results in worse OCR predictions using preparce_oc_image() and pad_image2() when preprocessing line images, although the results are visually and numerically almost identical. Refer to the Debug_OCR notebook to see the effect.
""" 

def preprocess_image_tf(image: np.array, target_height: int = 80, target_width: int = 2000):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=0.8, tileGridSize=(24,24))
    image = clahe.apply(image)
    tf_img = tf.expand_dims(image, axis=-1)
    tf_img = tf.image.resize_with_pad(tf_img, target_height, target_width)
    tf_img = tf_img / 255.0
    tf_img = tf.transpose(tf_img, perm=[1, 0, 2])
    tf_img = tf.squeeze(tf_img)
    
    return tf_img.numpy()
