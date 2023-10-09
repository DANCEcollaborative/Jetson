import os
import gc
import sys
from pathlib import Path
from typing import List, Union, Tuple
from collections import deque
import pickle

import cv2
import numpy as np
from PIL.Image import Image
import PIL.Image as img
from torch.profiler import profile, record_function, ProfilerActivity

import torch
from torch.cuda.amp import autocast
from facenet_pytorch import InceptionResnetV1, MTCNN
from facenet_pytorch.models.utils.detect_face import extract_face
sys.path.append(str(Path(__file__).resolve().parent.parent))
from confusion_model.data_utils import convert_from_image_to_cv2, timer_func, get_closest_ix, scale_box, create_yolo_tensor, post_process_yolo_preds
from confusion_model.constants import FACE_EMBEDDING_MODEL, EMOTION_NO
from confusion_model.tensorrt_conversion import load_model
from yolo_models.yolo import Model
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
        verbose: bool = False,
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
        self.model.to(device)
        self.data_type = data_type
        self.device = device
        self.verbose = verbose
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
        extractor: str = "fast",
        cv2_device: str = "cpu", 
        tensor_rt: bool = False
    ):
        super().__init__(load_model_path, data_type, label_dict, device)
        """
        If needed load CNN featurizer models for embedding
        """
        self.feat_type = load_model_path.split("/")[-1].split(".")[0].split("_")[-3]
        self.cv2_device = cv2_device
        self.extractor = extractor
        self.tensor_rt = tensor_rt
        
        # Default extraction, only works on newer cv2 releases
        if haar_path is None:
            haar_path = cv2.data.haarcascades + "haarcascade_frontalface_alt.xml"

        if self.extractor == "fast":
            # If running Haar Cascades on Cuda, will need to use cuda optimized classifier
            # Currently hard-coding Haar cascade Hyperparams
            if self.cv2_device == "cuda":
                self.face_extractor = cv2.cuda_CascadeClassifier.create(haar_path)
                self.face_extractor.setMinNeighbors(5)
                self.face_extractor.setMinObjectSize((10, 10))
            else:
                self.face_extractor = cv2.CascadeClassifier(haar_path)
        else:
            model_path = "/usr0/home/nvaikunt/Jetson/data/yolov5n-face_new.pt"
            cfg_dict_pth = "/usr0/home/nvaikunt/Jetson/data/yolov5n-cfg_dict.pkl"
            with open(cfg_dict_pth, 'rb') as handle:
                cfg_dict = pickle.load(handle)
            model = Model(cfg=cfg_dict)
            model.load_state_dict(torch.load(model_path))
            model.to(self.device)
            self.face_extractor = model.eval()
        
        # Load in CNN model and put on Cuda Device
        if self.verbose:
            start_mem, total_mem = torch.cuda.mem_get_info()
            print(f"Cuda usage before loading model {start_mem}")
            print(f"Cuda total mem: {total_mem}")
        if self.tensor_rt: 
            self.cnn = load_model("./data/Inception_Net_TRT_Window.pth")
            self.cnn.to(device=device)
        else: 
            with autocast():
                self.cnn = InceptionResnetV1(
                    pretrained=FACE_EMBEDDING_MODEL, classify=False, device=self.device
                )
                if self.verbose:
                    end_mem, total_mem = torch.cuda.mem_get_info()
                    print(f"Cuda total mem: {total_mem}")
                    print(f"Cuda usage before loading model {end_mem}")
        self.cnn.eval()

        # Type of Loaded Prediction Model was trained to perform
        self.multiclass = multiclass
        # Buffer of features
        self.feats = []

    def face_extraction_harr(self, image: Image) -> Union[List[Tuple[int, torch.Tensor]], None]:
        """
        Code for Haar Cascade based facial extraction. Currently used
        due to computational superiority
        :param image: PIL Image of Frame
        :return: Tensor of Size [Num Faces x 3 x 160 x 160]
        """
        # Take PIL image and turn it into CV2 image
        col_img, gray_img = convert_from_image_to_cv2(image, new_area=None)
        # If GPU, need to turn from numpy array to GPU Matrix and back
        if self.cv2_device == "cuda":
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
                gray_img, scaleFactor=1.1, minNeighbors=2, minSize=(10, 10)
            )
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
        faces = [extract_face(image, box) for box in boxes]
        left_bound = [box[0] for box in boxes]
        return list(zip(left_bound, faces))
    @timer_func
    def collate_frames(self, images: Union[Image, List[Image]]):
        """
        Take in a list of frames and extract faces from each frame. If all frames have the
        same amount of faces, then return a list of image tensors where each tensor
        is representative of a frame and the first dimension of the tensor is the face.
        If frames differ in face number, send to a function that heuristically aligns the faces
        and duplicates some faces on the time dimension
        :param images:
        :return:
        """
        if self.extractor == "fast":
            frame_list = [self.face_extraction_harr(image) for image in images if image is not None]
        else:
            # frame_list = [self._stable_facial_extraction(image) for image in images if image is not None]
            frame_list = [self._yolo_extraction(image) for image in images if image is not None]

        max_len, max_vector, max_ix = len(frame_list[0]), frame_list[0], 0
        need_match = False
        for i in range(1, len(frame_list)):
            curr_frame = frame_list[i]
            if len(curr_frame) != max_len:
                need_match = True
                if len(curr_frame) > max_len:
                    max_len = len(curr_frame)
                    max_vector = curr_frame
                    max_ix = i
        if need_match:
            frame_list = self.order_faces(frame_list, max_vector, max_ix)

        split_frames = [zip(*zipped_frames) for zipped_frames in frame_list]
        final_faces = [torch.stack(faces) for coordinates, faces in split_frames]
        return final_faces

    @staticmethod
    def order_faces(
        frame_list: List[List[Tuple[int, torch.Tensor]]],
        max_vector: List[Tuple[int, torch.Tensor]],
        max_ix: int,
    ) -> List[List[Tuple[int, torch.Tensor]]]:
        final_frames = []
        max_vector_coords = [coord for coord, face in max_vector]
        for i in range(len(frame_list)):
            if i == max_ix:
                final_frames.append(max_vector)
                continue
            curr_face = frame_list[i]
            closest_ix = {get_closest_ix(coord, max_vector_coords): i for i, (coord, face) in enumerate(curr_face)}
            aligned_face = []
            for i in range(len(max_vector)):
                if i not in closest_ix:
                    aligned_face.append(max_vector[i])
                else:
                    aligned_face.append(curr_face[closest_ix[i]])
            final_frames.append(aligned_face)
        return final_frames

    @timer_func
    def _stable_facial_extraction(self, image: Image, threshold: float = 0.95):
        """
        Code for MTCNN based facial extraction. Currently not supported due to
        computational constraints
        :param image: PIL Image of Frame
        :param threshold: Confidence Prob Threshold of MTCNN to include
        :return: Tensor of Size [Num Faces x 3 x 160 x 160]
        """
        start = time()
        boxes, probs = self.face_extractor.detect(image)
        print(f"MTCNN executed in {time() - start} s")
        boxes = np.array(
            [box for box, prob in zip(boxes, probs) if prob > threshold],
            dtype=np.float32,
        )
        if boxes.shape[0] == 0:
            return None
        boxes = boxes[boxes[:, 0].argsort()]
        faces = [extract_face(image, box) for box in boxes]
        left_bound = [box[0] for box in boxes]
        return list(zip(left_bound, faces))
    
    @timer_func
    def _yolo_extraction(self, image: Image, model_img_size = (640, 640), model_dtype = "f32"):
        """
        Code for MTCNN based facial extraction. Currently not supported due to
        computational constraints
        :param image: PIL Image of Frame
        :param threshold: Confidence Prob Threshold of MTCNN to include
        :return: Tensor of Size [Num Faces x 3 x 160 x 160]
        """
        if  model_dtype == "f16": 
            model_dtype = "f16"
        else: 
            model_dtype = "f32"
     
        input = create_yolo_tensor(image, model_img_size, model_dtype=model_dtype).to(self.device)
        with torch.no_grad():
            start = time()
            preds = self.face_extractor(input)
            if preds is None: 
                return None
            print(f"Yolo executed in {time() - start} s")
            boxes = post_process_yolo_preds(preds, image)
        if boxes.shape[0] == 0:
            return None
        boxes = boxes[boxes[:, 0].argsort()]
        faces = [extract_face(image, box) for box in boxes]
        left_bound = [box[0] for box in boxes]
        return list(zip(left_bound, faces))
            

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
            tensor_list = self.collate_frames(images)
            if len(tensor_list) != self.window_len:
                return None
            #TODO: Only Batch at the window level, change 287 to loop through faces 
            # -> this will allow us to use fixed batch size 
            try:
                full_tensor = torch.stack(tensor_list, dim=1)
            except RuntimeError:
                tensor_list = [torch.mean(tensor, dim=0) for tensor in tensor_list]
                full_tensor = torch.stack(tensor_list, dim=1)
            except TypeError:
                print("Face not detected in these frames!")
                full_tensor = None
            if full_tensor is not None:
                # Run through feature extractor
                if self.verbose:
                    start_mem, total_mem = torch.cuda.mem_get_info()
                    print(f"Cuda usage before loading tensor {start_mem}")
                with autocast():
                    start = time()
                    if self.tensor_rt:
                        full_tensor = full_tensor.half()
                    cnn_feats = torch.stack([self.cnn(tensor.to(self.device)) for tensor in full_tensor], dim=0)
                    print(f"CNN executed in {time() - start} s")
                if self.verbose:
                    print(f"CNN executed in {time() - start} s")
            else:
                return None
        return cnn_feats

    def add_image(self, image: Image):
        cnn_feats = self.extract_cnn_feats(image)
        if cnn_feats is not None:
            self.feats.append(cnn_feats)

    def is_ready(self) -> bool:
        return len(self.feats) == self.window_len

    @timer_func
    def run_inference(
        self,
        images: List[Image],
        output_format: str = "ix",
        threshold: Union[float, List[float]] = 0.6,
    ) -> Union[int, str, List[int], List[str]]:
        # Type check

        if self.data_type == "window" and len(images) != self.window_len:
            raise ValueError(f"Number of Images needs to be {self.window_len}")

        with torch.no_grad():
            features = self.extract_cnn_feats(images)
            if features is None: 
                return torch.tensor([0.0] * 5)
            features = features.reshape(-1, self.input_sz)
            if self.tensor_rt: 
                self.model = self.model.half()
            logits = self.model(features)
            if self.multiclass:
                preds = torch.argmax(logits, dim=-1)
                if output_format == "name":
                    preds = [self.ix_to_emotion[pred] for pred in preds]
            else:
                preds = torch.sigmoid(logits)
                print(preds)
        return preds


if __name__ == "__main__":
    # Inference check
    # model_path = "/home/teledia/Desktop/nvaikunt/ConfusionDataset/data/FCN_CNN_512_3.bin"
    model_path = (
        "/usr0/home/nvaikunt/FCN_CNN_512_3.bin"
    )
    # print(sys.path)
    model_path = "/usr0/home/nvaikunt/FCN_CNN_512_3.bin"
    torch.cuda.empty_cache()
    gc.collect()
    inference_model = ConfusionInference(
        load_model_path=model_path,
        data_type="window",
        multiclass=False,
        label_dict=EMOTION_NO,
        # haar_path="/home/teledia/Desktop/nvaikunt/ConfusionDataset/data/haarcascade_frontalface_alt_cuda.xml",
        device="cuda",
        haar_path=None,
        extractor="stable",
        tensor_rt = True
    )
    # file_path = "/home/teledia/Desktop/nvaikunt/ConfusionDataset/data/full_images"
    file_path = "/usr0/home/nvaikunt/full_images"
    #file_path = (
    #    "/Users/navaneethanvaikunthan/Documents/ConfusionDataset/data/full_images"
    #)
    dirlist = os.listdir(file_path)
    print(f"Number of images in buffer: {len(dirlist)}")
    buffer = [img.open(os.path.join(file_path, img_file)) for img_file in dirlist]
    buffer = deque(buffer)
    num_preds = 0
    window_len = inference_model.window_len
    start = time()
    while buffer:
        images = []
        image_1 = buffer.popleft()
        h, w = image_1.size
        #new_size = (int(.75 * w), int(.75 * h))
        new_size = (640, 640)
        image_1 = image_1
        images.append(image_1.resize(new_size))
        images.append(buffer.popleft().resize(new_size))
        images.append(buffer.popleft().resize(new_size))
        # Pseudo
        preds = inference_model.run_inference(images)
        num_preds += 1
    print(f"Total number of predictions {num_preds}")
    print(f"Total inference time: {time() - start}")
