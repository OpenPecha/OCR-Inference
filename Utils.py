import os
import cv2
import math
import numpy as np


def get_file_name(x):
    return os.path.basename(x).split(".")[0]


def create_dir(dir_name: str) -> None:
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def resize_image(orig_img: np.array, target_width: int = 2048) -> np.array:
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


def pad_image(img: np.array, patch_size: int = 64, is_mask=False) -> np.array:
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


def patch_image(
    img: np.array, patch_size: int = 64, overlap: int = 2, is_mask=False
) -> list:
    img, (pad_x, pad_y) = pad_image(img, patch_size, is_mask=is_mask)

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

    return patches, (pad_x, pad_y)


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


def optimize_countour(cnt, e=0.001):
    epsilon = e * cv2.arcLength(cnt, True)
    return cv2.approxPolyDP(cnt, epsilon, True)
