import cv2
import json
import logging
import numpy as np
import onnxruntime as ort
from scipy.special import expit
from scipy.signal import find_peaks
from fast_ctc_decode import viterbi_search
from Utils import (
    pad_image,
    patch_image,
    unpatch_image,
    resize_image,
    unpatch_prediction,
    rotate_page,
    prepare_ocr_image,
    preprocess_image_tf
)


class LineDetection:
    """
    Handles layout detection
    Args:
        - config_file: json file with the following parameters:
            - onnx model file
            - image input width and height
            - classes

    """

    def __init__(
        self,
        config_file: str,
        binarize_output: bool = False,
        dilate_kernel: int = 8,
        dilate_iterations: int = 6,
        execution_providers=None,
    ) -> None:
        self._config_file = config_file
        self._onnx_model_file = None
        self._patch_size = 512
        self._binarize_output = binarize_output
        self._dilate_kernel = dilate_kernel
        self._dilate_iterations = dilate_iterations
        self._inference = None
        # add other Execution Providers if applicable, see: https://onnxruntime.ai/docs/execution-providers
        if execution_providers is None:
            execution_providers = ["CPUExecutionProvider"]
        self.execution_providers = execution_providers

        self._init()

    def _init(self) -> None:
        _file = open(self._config_file)
        json_content = json.loads(_file.read())
        self._onnx_model_file = json_content["model"]
        self._patch_size = json_content["patch_size"]

        if self._onnx_model_file is not None:
            try:
                self._inference = ort.InferenceSession(
                    self._onnx_model_file, providers=self.execution_providers
                )
                self.can_run = True
            except Exception as error:
                logging.error(f"Error loading model file: {error}")
                self.can_run = False
        else:
            self.can_run = False

        logging.info(f"Line Inference -> Init(): {self.can_run}")

    def get_lines(self, image: np.array, prediction: np.array):
        line_contours, _ = cv2.findContours(
            prediction, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )

        x, y, w, h = cv2.boundingRect(prediction)

        if len(line_contours) == 0:
            return [], None, None

        elif len(line_contours) == 1:
            bbox_center = (x + w // 2, y + h // 2)
            peaks = [bbox_center]
            sorted_contours = {bbox_center: line_contours[0]}
            line_images = self.get_line_images(image, sorted_contours)

            return line_images, sorted_contours, peaks
        else:
            sorted_contours, peaks = self.sort_lines(prediction, line_contours)
            line_images = self.get_line_images(image, sorted_contours)

            return line_images, sorted_contours, peaks

    def sort_lines(self, line_prediction: np.array, contours: tuple):
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
                    peaks[(len(peaks) - 1) - idx]
                    - (peaks[(len(peaks) - 1) - (idx + 1)])
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

    def get_line_images(self, image: np.array, sorted_line_contours: dict):
        line_images = []

        for _, v in sorted_line_contours.items():
            image_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            cv2.drawContours(
                image_mask, [v], contourIdx=0, color=(255, 255, 255), thickness=-1
            )
            dilated1 = cv2.dilate(
                image_mask,
                kernel=(self._dilate_kernel, self._dilate_kernel),
                iterations=self._dilate_iterations,
                borderValue=0,
                anchor=(-1, 0),
                borderType=cv2.BORDER_DEFAULT,
            )
            dilated2 = cv2.dilate(
                image_mask,
                kernel=(self._dilate_kernel, self._dilate_kernel),
                iterations=self._dilate_iterations,
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

            if self._binarize_output:
                indices = np.where(cropped_img[:, :, 1] == 0)
                clear = cropped_img.copy()
                clear[indices[0], indices[1], :] = [255, 255, 255]
                clear_bw = cv2.cvtColor(clear, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(clear_bw, 170, 255, cv2.THRESH_BINARY)

                line_images.append(thresh)
            else:
                line_images.append(cropped_img)

        return line_images

    def generate_line_images(self, image: np.array, prediction: np.array):
        """
        Applies some rotation correction to the original image and creates the line images based on the predicted lines.
        """
        rotated_img, rotated_prediction, angle = rotate_page(
            original_image=image, line_mask=prediction
        )
        line_images, sorted_contours, peaks = self.get_lines(
            rotated_img, rotated_prediction
        )

        return line_images, sorted_contours, peaks

    def predict(
        self,
        original_image: np.array,
        unpatch_type: int = 0,
        class_threshold: float = 0.8,
    ) -> np.array:
        image, _ = resize_image(original_image)
        padded_img, (pad_x, pad_y) = pad_image(image, self._patch_size)
        image_patches, y_splits = patch_image(padded_img, self._patch_size)
        image_batch = np.array(image_patches)
        image_batch = image_batch.astype(np.float32)
        image_batch /= 255.0

        image_batch = np.transpose(image_batch, axes=[0, 3, 1, 2])  # make B x C x H xW

        ort_batch = ort.OrtValue.ortvalue_from_numpy(image_batch)
        ocr_results = self._inference.run_with_ort_values(
            ["output"], {"input": ort_batch}
        )
        prediction = ocr_results[0].numpy()
        prediction = np.squeeze(prediction, axis=1)
        prediction = expit(prediction)
        prediction = np.where(prediction > class_threshold, 1.0, 0.0)
        pred_list = [prediction[x, :, :] for x in range(prediction.shape[0])]

        if unpatch_type == 0:
            unpatched_image = unpatch_image(image, pred_list)
        else:
            unpatched_image = unpatch_prediction(image, y_splits)

        cropped_image = unpatched_image[
            : unpatched_image.shape[0] - pad_y, : unpatched_image.shape[1] - pad_x
        ]
        # back_sized = cv2.resize(cropped_image, (int(cropped_image.shape[1] / resize_factor), int(cropped_image.shape[0] / resize_factor)))
        back_sized_image = cv2.resize(
            cropped_image, (original_image.shape[1], original_image.shape[0])
        )
        back_sized_image = back_sized_image.astype(np.uint8)

        line_images, sorted_contours, peaks = self.generate_line_images(
            original_image, back_sized_image
        )
        return back_sized_image, line_images, sorted_contours, peaks


class OCRInference:
    def __init__(self, config_file) -> None:
        self.config = config_file
        self._onnx_model_file = None
        self._input_width = 2000
        self._input_height = 80
        self._characters = []
        self._can_run = False
        self.ocr_session = None

        self._init()

    def _init(self) -> None:
        _file = open(self.config, encoding="utf-8")
        json_content = json.loads(_file.read())
        self._onnx_model_file = json_content["model"]
        self._input_width = json_content["input_width"]
        self._input_height = json_content["input_height"]
        self.charset = self.get_model_characters(json_content["charset"])
        self.ocr_session = ort.InferenceSession(
            self._onnx_model_file, providers=["CPUExecutionProvider"]
        )

    def get_model_characters(self, characters):
        characters = [c for c in characters]
        characters.append("[BLK]")
        characters.insert(0, "[UNK]")

        return characters

    def run(self, line_images: list, replace_blank: str = ""):
        """
        See comment in Utils.py. So preprocess_image_tf() is used for time being.
        img_batch = [prepare_ocr_image(x) for x in line_images]
        """
        img_batch = [preprocess_image_tf(x) for x in line_images]
        img_batch = np.array(img_batch)
        ort_batch = ort.OrtValue.ortvalue_from_numpy(img_batch)
        ocr_results = self.ocr_session.run_with_ort_values(
            ["Final"], {"the_input": ort_batch}
        )
        prediction = ocr_results[0].numpy()

        predicted_text = []
        for idx in range(prediction.shape[0]):
            pred_line = prediction[idx, :, :]
            text, _ = viterbi_search(pred_line, self.charset)
            text = text.replace("[BLK]", replace_blank)

            if len(text) > 0:
                predicted_text.append(text)
            else:
                predicted_text.append("")

        return predicted_text, prediction
