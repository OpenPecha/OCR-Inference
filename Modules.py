import cv2
import json
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
    generate_line_images,
    prepare_ocr_image,
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
        mode: str = "cpu",
    ) -> None:
        self._config_file = config_file
        self._onnx_model_file = None
        self._patch_size = 512
        self._binarize_output = binarize_output
        self._inference = None
        # add other Execution Providers if applicable, see: https://onnxruntime.ai/docs/execution-providers
        self.mode = mode
        
        if self.mode == "cuda":
            execution_providers = ["CUDAExecutionProvider"]
        else:
            execution_providers = ["CPUExecutionProvider"]

        self.execution_providers = execution_providers

        self._init()

    def _init(self) -> None:
        _file = open(self._config_file)
        json_content = json.loads(_file.read())
        
        if self.mode == "cuda":
            self._onnx_model_file = json_content["gpu-model"]
        else:
            self._onnx_model_file = json_content["cpu-model"]
 
        self._patch_size = json_content["patch_size"]


        if self._onnx_model_file is not None:
            try:
                self._inference = ort.InferenceSession(
                    self._onnx_model_file, providers=self.execution_providers
                )
                self.can_run = True
            except Exception as error:
                print.error(f"Error loading model file: {error}")
                self.can_run = False
        else:
            self.can_run = False

        print(f"Line Inference -> Init(): {self.can_run}")

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
        prediction = self._inference.run_with_ort_values(
            ["output"], {"input": ort_batch}
        )
        prediction = prediction[0].numpy()
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

        # TODO: remove this into a post-processing module
        line_images, sorted_contours, bbox, peaks = generate_line_images(
            original_image, back_sized_image
        )
        return back_sized_image, line_images, sorted_contours, bbox, peaks
    

    def predict_batch(self, image_list: list[np.array]):
        pass



class OCRInference:
    def __init__(self, config_file, mode: str = "cpu") -> None:
        self.config = config_file
        self._onnx_model_file = None
        self._input_width = 2000
        self._input_height = 80
        self._characters = []
        self._can_run = False
        self.ocr_session = None
        self.mode = mode
        self._init()

    def _init(self) -> None:
        _file = open(self.config, encoding="utf-8")
        json_content = json.loads(_file.read())

        if self.mode == "cuda":
            self._onnx_model_file = json_content["gpu-model"]
        else:
            self._onnx_model_file = json_content["cpu-model"]

        self._input_width = json_content["input_width"]
        self._input_height = json_content["input_height"]
        self._input_layer = json_content["input_layer"]
        self._output_layer = json_content["output_layer"]
        self._add_channel_dim = json_content["add_channel_dim"]
        self.charset = json_content["charset"]

        if self.mode == "cuda":
            execution_providers = ["CUDAExecutionProvider"]
        else:
            execution_providers = ["CPUExecutionProvider"]
        self.ocr_session = ort.InferenceSession(
            self._onnx_model_file, providers=execution_providers
        )

    def run(self, line_images: list, replace_blank: str = ""):
        img_batch = [
            prepare_ocr_image(
                x, target_width=self._input_width, target_height=self._input_height
            )
            for x in line_images
        ]
        img_batch = np.array(img_batch)

        ort_batch = ort.OrtValue.ortvalue_from_numpy(img_batch)
        ocr_results = self.ocr_session.run_with_ort_values(
            [self._output_layer], {self._input_layer: ort_batch}
        )
        prediction = ocr_results[0].numpy()
        prediction = np.transpose(prediction, axes=[1, 0, 2])
        # print(f"Raw prediction: {prediction.shape}")

        predicted_text = []
        for idx in range(prediction.shape[0]):
            pred_line = prediction[idx, :, :]
            text, _ = viterbi_search(pred_line, self.charset)

            if len(text) > 0:
                predicted_text.append(text)
            else:
                predicted_text.append("")

        return predicted_text, prediction
