import cv2
import json
import logging
import numpy as np
import onnxruntime as ort
from scipy.special import expit
from fast_ctc_decode import viterbi_search
from Utils import patch_image, unpatch_image, resize_image


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
        execution_providers=None,
    ) -> None:
        self._config_file = config_file
        self._onnx_model_file = None
        self._patch_size = 512
        self._can_run = False
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

    def predict(self, original_image: np.array) -> np.array:
        image, resize_factor = resize_image(original_image)
        image_patches, (pad_x, pad_y) = patch_image(image, self._patch_size)
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
        prediction = np.where(prediction > 0.8, 1.0, 0.0)
        pred_list = [prediction[x, :, :] for x in range(prediction.shape[0])]

        unpatched_image = unpatch_image(image, pred_list)
        cropped_image = unpatched_image[
            : unpatched_image.shape[0] - pad_y, : unpatched_image.shape[1] - pad_x
        ]
        # back_sized = cv2.resize(cropped_image, (int(cropped_image.shape[1] / resize_factor), int(cropped_image.shape[0] / resize_factor)))
        back_sized = cv2.resize(
            cropped_image, (original_image.shape[1], original_image.shape[0])
        )
        back_sized = back_sized.astype(np.uint8)
        back_sized = cv2.cvtColor(back_sized, cv2.COLOR_GRAY2BGR)

        return back_sized