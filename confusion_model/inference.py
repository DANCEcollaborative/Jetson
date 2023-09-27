import os
from typing import List, Union
from collections import deque

import cv2
from PIL.Image import Image
import PIL.Image as img

import torch
from facenet_pytorch import InceptionResnetV1
from facenet_pytorch.models.utils.detect_face import extract_face

from data_utils import convert_from_image_to_cv2, timer_func
from constants import FACE_EMBEDDING_MODEL, EMOTION_NO
from time import time


class ConfusionInferenceBase:
    """
    Base class for Inference, contains shared init code
    wherein model is loaded
    Later maybe shared methods to be implemented or overriden will be introduced
    """

    def __init__(
            self,
            load_model_path: str,
            data_type: str = "window",
            label_dict: dict = EMOTION_NO,
            device: str = "cpu",
    ):
        """
        Initialize trained model for inference
        :param load_model_path: Path to load PyTorch model from
        :param label_dict: Dictionary of emotion labels
        :param data_type: What kind of data model was trained with
        :param device: Device for model
        """
        self.load_model_path = load_model_path
        # Extract labels in order to go between index and label preds
        self.label_dict = label_dict
        self.ix_to_emotion = {ix: label for label, ix in label_dict.items()}

        # Extract Model Type from File path
        self.model_type = load_model_path.split("/")[-1].split(".")[0].split("_")[0]
        self.input_sz = int(load_model_path.split("/")[-1].split(".")[0].split("_")[-2])
        self.window_len = int(
            load_model_path.split("/")[-1].split(".")[0].split("_")[-1]
        )
        # Load model and put on device
        self.model = torch.load(self.load_model_path)
        self.model.eval()
        self.model.to("cpu")
        self.data_type = data_type
        self.device = device

        if data_type == "window":
            self.input_sz *= self.window_len


class ConfusionInference(ConfusionInferenceBase):
    """
    Execute and time single frame visual emotion detection inference,
    may include batch support features
    """

    def __init__(
            self,
            load_model_path: str,
            data_type: str = "window",
            label_dict: dict = EMOTION_NO,
            device: str = "cpu",
            multiclass: bool = False,
            haar_path: str = None,
    ):
        super().__init__(load_model_path, data_type, label_dict, device)
        """
        If needed load CNN featurizer models for embedding
        """
        self.feat_type = load_model_path.split("/")[-1].split(".")[0].split("_")[-3]
        if self.feat_type == "CNN":
            # Default extraction, only works on newer cv2 releases
            if haar_path is None:
                haar_path = cv2.data.haarcascades + "haarcascade_frontalface_alt.xml"

            # If running Haar Cascades on Cuda, will need to use cuda optimized classifier
            # Currently hard-coding Haar cascade Hyperparams
            if self.device == "cuda":
                self.face_extractor = cv2.cuda_CascadeClassifier.create(haar_path)
                self.face_extractor.setMinNeighbors(7)
                self.face_extractor.setMinObjectSize((10,10))
            else:
                self.face_extractor = cv2.CascadeClassifier(haar_path)

            # Load in CNN model and put on Cuda Device
            self.cnn = InceptionResnetV1(
                pretrained=FACE_EMBEDDING_MODEL, classify=False
            ).to(device)
            self.cnn.eval()

        # Type of Loaded Prediction Model was trained to perform
        self.multiclass = multiclass
        # Buffer of features
        self.feats = []

    @timer_func
    def _face_extraction_harr(self, image: Image):
        """
        Code for Haar Cascade based facial extraction. Currently used
        due to computational superiority
        :param image: PIL Image of Frame
        :return: Tensor of Size [Num Faces x 3 x 160 x 160]
        """
        # Take PIL image and turn it into CV2 image
        col_img, gray_img = convert_from_image_to_cv2(image, new_area=None)
        # If GPU, need to turn from numpy array to GPU Matrix and back
        if self.device == "cuda":
            cuFrame = cv2.cuda_GpuMat(gray_img)
            boxes = self.face_extractor.detectMultiScale(cuFrame).download()
            # Given we return anything, then unpack the value
            if boxes is not None:
                boxes = boxes[0]
            else:
                return None
        else:
            # CPU code has same logic but with different API and I/O
            boxes = self.face_extractor.detectMultiScale(
            gray_img, scaleFactor=1.1, minNeighbors=7, minSize=(10, 10))
        if len(boxes) == 0:
            return None
        # If results are returned, change boxes from (x, y, h, w) to
        # (x1, y1, x2, y2) -> as this is expected
        boxes = boxes[boxes[:, 0].argsort()]
        boxes[:, 2] = boxes[:, 2] + boxes[:, 0]
        boxes[:, 3] = boxes[:, 3] + boxes[:, 1]

        """
        #Test bounding boxes
        for (x, y, x2, y2) in boxes:
            cv2.rectangle(col_img, (x, y), (x2, y + y2), (0, 255, 0), 4)
        cv2.imshow("Display", col_img)
        cv2.waitKey(0)
        """
        # Return extracted faces
        return torch.stack([extract_face(image, box) for box in boxes])

    @timer_func
    def extract_cnn_feats(
            self, images: Union[Image, List[Image]], threshold: float = 0.95
    ):
        """
        Take in image frame and extract + embed all faces present
        :param images: Frames in PIL.Image format or group of frames
        :param threshold: How confident that it's a human face
        :return: CNN based features for Confusion Detection Model
        """

        with torch.no_grad():
            # Get Embedded Face Images
            tensor = self._face_extraction_harr(images)
            if tensor is not None:

                start = time()
                # Run through feature extractor
                cnn_feats = self.cnn(tensor.reshape(-1, 3, 160, 160).to(self.device))
                print(f"CNN executed in {time() - start} s")
            else:
                return None
        return cnn_feats.cpu()

    @timer_func
    def run_inference(
            self,
            output_format: str = "ix",
            threshold: Union[float, List[float]] = 0.6,
    ) -> Union[int, str, List[int], List[str]]:
        # Type check

        if self.data_type == "window":
            try:
                # If each frame in the window has the same amount of extracted
                # faces, then stack them into 1 tensor
                features = torch.stack(self.feats, dim=1)
            except RuntimeError:
                # Otherwise take average over faces
                features = torch.stack([torch.mean(feat, dim=0) for feat in self.feats], dim=1)
        else:
            # If not windowed, then use last frame available
            features = self.feats[-1]

        with torch.no_grad():
            features = features.reshape(-1, self.input_sz)
            logits = self.model(features)
            if self.multiclass:
                preds = torch.argmax(logits, dim=-1)
                if output_format == "name":
                    preds = [self.ix_to_emotion[pred] for pred in preds]
            else:
                preds = torch.sigmoid(logits)
                print(preds)
                preds = torch.gt(preds, threshold)

        return preds


if __name__ == "__main__":
    # extract_images_from_vid("data/videos/confusion_subset/giphy.mp4")
    # Inference check
    inference_model = ConfusionInference(
        load_model_path="./data/FCN_CNN_512_3.bin",
        data_type="window",
        multiclass=False,
        label_dict=EMOTION_NO,
        device="cpu",
        haar_path=None
    )
    dirlist = os.listdir("data/full_images")
    buffer = [img.open(os.path.join("data/full_images", img_file)) for img_file in dirlist]
    buffer = deque(buffer)
    window_len = inference_model.window_len
    while buffer:
        curr_image = buffer.popleft()
        h, w = curr_image.size
        curr_image = curr_image.resize((3 * h // 4,  3 * w // 4))
        print(curr_image.size)
        feat_tensor = inference_model.extract_cnn_feats(curr_image)
        if feat_tensor is not None:
            inference_model.feats.append(feat_tensor)
        if len(inference_model.feats) == window_len:
            preds = inference_model.run_inference()
            print(preds)
            inference_model.feats.pop(0)