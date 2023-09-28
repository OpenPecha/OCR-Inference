import os
import cv2
import math
import numpy as np
import tensorflow as tf
from typing import Tuple
from datetime import datetime
from xml.dom import minidom
from scipy.signal import find_peaks
import xml.etree.ElementTree as etree


def get_file_name(x) -> str:
    return os.path.basename(x).split(".")[0]


def read_label(gt_file: str) -> str:
    f = open(gt_file, "r", encoding="utf-8")
    return f.readline()


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
    image = cv2.resize(
        image,
        (int(image.shape[1] * width_ratio), target_height),
        interpolation=cv2.INTER_LINEAR,
    )
    return image


def resize_to_width(image, target_width: int):
    width_ratio = target_width / image.shape[1]
    image = cv2.resize(
        image,
        (target_width, int(image.shape[0] * width_ratio)),
        interpolation=cv2.INTER_LINEAR,
    )
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

    return cv2.resize(
        out_img, (target_width, target_height), interpolation=cv2.INTER_LINEAR
    )


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
    mask_threshold = (line_mask.shape[0] * line_mask.shape[1]) * 0.001
    contours = [x for x in contours if cv2.contourArea(x) > mask_threshold]
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
        # print(f"Defaulting to 0: {angles}")
        mean_angle = 0

    # print(f"Rotation angle: {mean_angle}")

    rows, cols = original_image.shape[:2]
    rot_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), mean_angle, 1)
    rotated_img = cv2.warpAffine(
        original_image, rot_matrix, (cols, rows), borderValue=(0, 0, 0)
    )

    rotated_prediction = cv2.warpAffine(
        line_mask, rot_matrix, (cols, rows), borderValue=(0, 0, 0)
    )

    return rotated_img, rotated_prediction, mean_angle


def get_line_images(
    image: np.array,
    sorted_line_contours: dict,
    dilate_kernel: int = 20,
    binarize: bool = False,
):
    line_images = []

    for _, contour in sorted_line_contours.items():
        image_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        cv2.drawContours(
            image_mask, [contour], contourIdx=0, color=(255, 255, 255), thickness=-1
        )

        _, _, _, height = cv2.boundingRect(contour)

        if height < 60:
            dilate_iterations = dilate_kernel * 1
        elif height >= 60 and height < 90:
            dilate_iterations = dilate_kernel * 2
        else:
            dilate_iterations = dilate_kernel * 6

        dilated1 = cv2.dilate(
            image_mask,
            kernel=(dilate_kernel, dilate_kernel),
            iterations=dilate_iterations,
            borderValue=0,
            anchor=(-1, 0),
            borderType=cv2.BORDER_DEFAULT,
        )
        dilated2 = cv2.dilate(
            image_mask,
            kernel=(dilate_kernel, dilate_kernel),
            iterations=dilate_iterations,
            borderValue=0,
            anchor=(0, 1),
            borderType=cv2.BORDER_DEFAULT,
        )
        combined = cv2.add(dilated1, dilated2)
        image_masked = cv2.bitwise_and(image, image, mask=combined)

        cropped_img = np.delete(
            image_masked, np.where(~image_masked.any(axis=1))[0], axis=0
        )
        cropped_img = np.delete(
            cropped_img, np.where(~cropped_img.any(axis=0))[0], axis=1
        )

        if binarize:
            indices = np.where(cropped_img[:, :, 1] == 0)
            clear = cropped_img.copy()
            clear[indices[0], indices[1], :] = [255, 255, 255]
            clear_bw = cv2.cvtColor(clear, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(clear_bw, 170, 255, cv2.THRESH_BINARY)

            line_images.append(thresh)
        else:
            line_images.append(cropped_img)

    return line_images


def get_lines(image: np.array, prediction: np.array):
    line_contours, _ = cv2.findContours(
        prediction, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    x, y, w, h = cv2.boundingRect(prediction)

    if len(line_contours) == 0:
        return [], None, None, None

    elif len(line_contours) == 1:
        bbox_center = (x + w // 2, y + h // 2)
        peaks = [bbox_center]
        sorted_contours = {bbox_center: line_contours[0]}
        line_images = get_line_images(image, sorted_contours)

        return line_images, sorted_contours, (x, y, w, h), peaks
    else:
        sorted_contours, peaks = sort_lines(prediction, line_contours)
        line_images = get_line_images(image, sorted_contours)

        return line_images, sorted_contours, (x, y, w, h), peaks


def sort_lines(line_prediction: np.array, contours: tuple):
    """
    A preliminary approach to sort the found contours and sort them by reading lines. The relative distance between the lines is currently taken as roughly constant,
    wherefore mean // 2 is taken as threshold for line breaks. This might not work in scenarios in which the line distances are less constant.

    Args:
        - tuple of contours returned by cv2.findContours()
    Returns:
        - dictionary of {(bboxcenter_x, bbox_center_y) : [contour]}
        - peaks returned by find_peaks() marking the line breaks
    """

    horizontal_projection = np.sum(line_prediction, axis=1)
    horizontal_projection = horizontal_projection / 255
    mean = int(np.mean(horizontal_projection))
    peaks, _ = find_peaks(horizontal_projection, height=mean, width=4)

    # calculate the line distances
    line_distances = []
    for idx in range(len(peaks)):
        if idx < len(peaks) - 1:
            line_distances.append(
                peaks[(len(peaks) - 1) - idx] - (peaks[(len(peaks) - 1) - (idx + 1)])
            )

    if len(line_distances) == 0:
        line_distance = 0
    else:
        line_distance = int(
            np.mean(line_distances)
        )  # that might not work great if the line distances are varying a lot

    # get the bbox centers of each contour and keep a reference to the contour in contour_dict
    centers = []
    contour_dict = {}

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        y_center = y + (h // 2)
        x_center = x + (w // 2)
        centers.append((x_center, y_center))
        contour_dict[(x_center, y_center)] = contour

    centers = sorted(centers, key=lambda x: x[1])

    # associate bbox centers with the peaks (i.e. line breaks)
    cnt_dict = {}

    for center in centers:
        if center == centers[-1]:
            cnt_dict[center[1]] = [center]
            continue

        for peak in peaks:
            diff = abs(center[1] - peak)
            if diff <= line_distance // 2:
                if peak in cnt_dict.keys():
                    cnt_dict[peak].append(center)
                else:
                    cnt_dict[peak] = [center]

    # sort bbox centers for x value to get proper reading order
    for k, v in cnt_dict.items():
        if len(v) > 1:
            v = sorted(v)
            cnt_dict[k] = v

    # build final dictionary with correctly sorted bbox_centers by y and x -> contour
    sorted_contour_dict = {}
    for k, v in cnt_dict.items():
        for l in v:
            sorted_contour_dict[l] = contour_dict[l]

    return sorted_contour_dict, peaks


def get_line_images(
    image: np.array,
    sorted_line_contours: dict,
    dilate_kernel: int = 20,
    binarize: bool = False,
):
    line_images = []

    for _, contour in sorted_line_contours.items():
        image_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        cv2.drawContours(
            image_mask, [contour], contourIdx=0, color=(255, 255, 255), thickness=-1
        )

        _, _, _, height = cv2.boundingRect(contour)

        if height < 60:
            dilate_iterations = dilate_kernel * 1
        elif height >= 60 and height < 90:
            dilate_iterations = dilate_kernel * 2
        else:
            dilate_iterations = dilate_kernel * 6

        dilated1 = cv2.dilate(
            image_mask,
            kernel=(dilate_kernel, dilate_kernel),
            iterations=dilate_iterations,
            borderValue=0,
            anchor=(-1, 0),
            borderType=cv2.BORDER_DEFAULT,
        )
        dilated2 = cv2.dilate(
            image_mask,
            kernel=(dilate_kernel, dilate_kernel),
            iterations=dilate_iterations,
            borderValue=0,
            anchor=(0, 1),
            borderType=cv2.BORDER_DEFAULT,
        )
        combined = cv2.add(dilated1, dilated2)
        image_masked = cv2.bitwise_and(image, image, mask=combined)

        cropped_img = np.delete(
            image_masked, np.where(~image_masked.any(axis=1))[0], axis=0
        )
        cropped_img = np.delete(
            cropped_img, np.where(~cropped_img.any(axis=0))[0], axis=1
        )

        if binarize:
            indices = np.where(cropped_img[:, :, 1] == 0)
            clear = cropped_img.copy()
            clear[indices[0], indices[1], :] = [255, 255, 255]
            clear_bw = cv2.cvtColor(clear, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(clear_bw, 170, 255, cv2.THRESH_BINARY)

            line_images.append(thresh)
        else:
            line_images.append(cropped_img)

    return line_images


def generate_line_images(image: np.array, prediction: np.array):
    """
    Applies some rotation correction to the original image and creates the line images based on the predicted lines.
    """
    rotated_img, rotated_prediction, angle = rotate_page(
        original_image=image, line_mask=prediction
    )
    line_images, sorted_contours, bbox, peaks = get_lines(
        rotated_img, rotated_prediction
    )

    return line_images, sorted_contours, bbox, peaks


def optimize_countour(cnt, e=0.001):
    epsilon = e * cv2.arcLength(cnt, True)
    return cv2.approxPolyDP(cnt, epsilon, True)


def prepare_ocr_image(
    line_image: np.array, target_width: int = 2000, target_height: int = 80
):
    if len(line_image.shape) > 2:
        line_image = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY)

    image = pad_image2(
        line_image,
        target_width=target_width,
        target_height=target_height,
        padding="white",
    )
    image = image.reshape((1, target_height, target_width))
    image = (image / 127.5) - 1.0
    image = image.astype(np.float32)
    return image


def get_utc_time():
    t = datetime.now()
    s = t.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    s = s.split(" ")

    return f"{s[0]}T{s[1]}"


def get_text_points(contour):
    points = ""
    for box in contour:
        point = f"{box[0][0]},{box[0][1]} "
        points += point
    return points


def get_text_line_block(coordinate, index, unicode_text):
    text_line = etree.Element(
        "Textline", id="", custom=f"readingOrder {{index:{index};}}"
    )
    text_line = etree.Element("TextLine")
    text_line_coords = coordinate

    text_line.attrib["id"] = f"line_9874_{str(index)}"
    text_line.attrib["custom"] = f"readingOrder {{index: {str(index)};}}"

    coords_points = etree.SubElement(text_line, "Coords")
    coords_points.attrib["points"] = text_line_coords
    text_equiv = etree.SubElement(text_line, "TextEquiv")
    unicode_field = etree.SubElement(text_equiv, "Unicode")
    unicode_field.text = unicode_text

    return text_line


def build_xml_document(
    image: np.array,
    image_name: str,
    text_region_bbox,
    coordinates: list,
    text_lines: list,
):
    root = etree.Element("PcGts")
    root.attrib[
        "xmlns"
    ] = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
    root.attrib["xmlns:xsi"] = "http://www.w3.org/2001/XMLSchema-instance"
    root.attrib[
        "xsi:schemaLocation"
    ] = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15 http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15/pagecontent.xsd"

    metadata = etree.SubElement(root, "Metadata")
    creator = etree.SubElement(metadata, "Creator")
    creator.text = "Transkribus"
    created = etree.SubElement(metadata, "Created")
    created.text = get_utc_time()

    page = etree.SubElement(root, "Page")
    page.attrib["imageFilename"] = image_name
    page.attrib["imageWidth"] = f"{image.shape[1]}"
    page.attrib["imageHeight"] = f"{image.shape[0]}"

    reading_order = etree.SubElement(page, "ReadingOrder")
    ordered_group = etree.SubElement(reading_order, "OrderedGroup")
    ordered_group.attrib["id"] = f"1234_{0}"
    ordered_group.attrib["caption"] = "Regions reading order"

    region_ref_indexed = etree.SubElement(reading_order, "RegionRefIndexed")
    region_ref_indexed.attrib["index"] = "0"
    region_ref = "region_main"
    region_ref_indexed.attrib["regionRef"] = region_ref

    text_region = etree.SubElement(page, "TextRegion")
    text_region.attrib["id"] = region_ref
    text_region.attrib["custom"] = "readingOrder {index:0;}"

    text_region_coords = etree.SubElement(text_region, "Coords")
    text_region_coords.attrib["points"] = text_region_bbox

    # print(f"Adding Text to Document: {len(coordinates)}, {len(text_lines)}")
    for i in range(0, len(coordinates)):
        text_coords = get_text_points(coordinates[i])
        # print(f"Unicode: {text_lines[i]}")
        if text_lines != None:
            text_region.append(
                get_text_line_block(text_coords, i, unicode_text=text_lines[i])
            )
        else:
            text_region.append(get_text_line_block(text_coords, i, unicode_text=""))

    xmlparse = minidom.parseString(etree.tostring(root))
    prettyxml = xmlparse.toprettyxml()

    return prettyxml
