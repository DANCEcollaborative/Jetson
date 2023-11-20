from argparse import Namespace
from .models import FullImageModel, KeypointsModel, OnePassModel
import math
import torch.nn as nn
from typing import List, Tuple
import torch
from typing import Union

import PIL.Image as PIL_Image
from PIL.Image import Image
import numpy as np
from yolo_models.yolo import Model
from torchvision.transforms import v2
from torchvision.models.detection import KeypointRCNN
from torchvision.models.detection.ssd import SSD
from yolo_utils.general import non_max_suppression_face

def get_cropped_image(
    image: Union[Image, str], bbox: Union[list, torch.Tensor, np.ndarray]
) -> Image:
    if not isinstance(image, Image):
        image = PIL_Image.open(image)

    if not isinstance(bbox, list):
        bbox = bbox.tolist()
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image.crop(tuple(bbox))


def get_hidden_layer_sizes(start_val, num_layers, strategy="bottleneck"):
    hidden_layers = []
    for i in range(num_layers):
        if strategy == "bottleneck":
            current_layer_sz = start_val // (i + 1)
        elif strategy == "expand":
            current_layer_sz = int(start_val * math.pow(1.1, (i + 1)))
        else:
            current_layer_sz = start_val
        hidden_layers.append(current_layer_sz)
    return hidden_layers


def load_model(model_args: Namespace) -> nn.Module:
    if model_args.model_type == "keypoints":
        model = KeypointsModel(
            num_hidden_fusion_layers=model_args.num_fusion_layers,
            dropout_p=model_args.dropout,
            hidden_size_const=model_args.hidden_sz_const,
            post_concat_feat_sz=model_args.post_concat_feat_sz,
        )
    else:
        if model_args.hidden_sz_strat == "constant":
            hidden_size_list = None
        else:
            hidden_size_list = get_hidden_layer_sizes(
                model_args.hidden_sz_const,
                model_args.num_fusion_layers,
                strategy=model_args.hidden_sz_strat,
            )
        if model_args.model_type == "full_image":
            model = FullImageModel(
                num_hidden_fusion_layers=model_args.num_fusion_layers,
                dropout_p=model_args.dropout,
                hidden_sizes=hidden_size_list,
                hidden_size_const=model_args.hidden_sz_const,
                post_concat_feat_sz=model_args.post_concat_feat_sz,
            )
        else:
            model = OnePassModel(
                num_hidden_layers=model_args.num_fusion_layers,
                dropout_p=model_args.dropout,
                hidden_sizes=hidden_size_list,
                hidden_size_const=model_args.hidden_sz_const,
            )
    return model



def filter_model_params(model: nn.Module, modules_to_filter: List = None):
    if modules_to_filter is None:
        modules_to_filter = ["regression.weight", "regression.bias"]
    filtered_params = list(
        filter(lambda kv: kv[0] in modules_to_filter, model.named_parameters())
    )
    base_params = list(
        filter(lambda kv: kv[0] not in modules_to_filter, model.named_parameters())
    )
    filtered_params = [param for _, param in filtered_params]
    base_params = [param for _, param in base_params]
    return filtered_params, base_params

def generate_face_bbox(
    model: Model,
    image: Union[Image, str],
    transforms: v2.Compose,
    device: str,
    conf_thresh: float = 0.7,
) -> np.ndarray:
    input_img = image.resize((640, 640))
    input_tens = transforms(input_img)
    input_tens = input_tens.to(device)
    if len(input_tens.shape) == 3:
        input_tens = input_tens.unsqueeze(0)
    with torch.no_grad():
        output = model(input_tens)
        preds = post_process_yolo_preds(output, image, conf_thres=conf_thresh)
    if len(preds) == 0:
        return []
    return preds[0]


def generate_person_bboxes(
    model: SSD,
    image: Union[Image, str],
    transforms: v2.Compose,
    device: str,
    detect_threshold: int = 0.5,
) -> list:
    input_tens = transforms(image)

    input_tens = input_tens.to(device)
    if len(input_tens.shape) == 3:
        input_tens = input_tens.unsqueeze(0)
    with torch.no_grad():
        output = model(input_tens)
        bxs = output[0]["boxes"]
        scores = output[0]["scores"]

        # Filter boxes by scores and labels
        idx = torch.where(scores > detect_threshold)
        classes = output[0]["labels"][idx]
        all_boxes = bxs[idx]
        boxes = [
            box for box, class_ix in zip(all_boxes, classes) if class_ix.item() == 1
        ]
        if len(boxes) == 0:
            return []
        boxes = torch.stack(boxes)
        return boxes.tolist()

def scale_box(img2_size, box, img1_size):
    h1, w1 = img1_size
    h2, w2 = img2_size
    x_scale = w1 / w2
    y_scale = h1 / h2

    box[:, [0, 2]] *= x_scale
    box[:, [1, 3]] *= y_scale
    return box

def generate_keypoints(
    model: KeypointRCNN,
    image: Union[Image, str],
    transforms: v2.Compose,
    device,
    detect_threshold=0.9,
    inference: bool = False,
) -> torch.Tensor:
    input_tens = transforms(image)
    input_tens = input_tens.to(device)
    if len(input_tens.shape) == 3:
        input_tens = input_tens.unsqueeze(0)
    with torch.no_grad():
        output = model(input_tens)

    # Filter out Keypoints for Objects w/ low prob of being human
    scores = output[0]["scores"]
    kpts = output[0]["keypoints"]
    idx = torch.where(scores > detect_threshold)
    keypoints = kpts[idx]

    # Keep only upper body keypoints
    keypoints = keypoints[:, :11, :]
    if inference:
        return keypoints
    keypoints = keypoints[keypoints[:, 0, 0].argsort()]
    return keypoints
def post_process_yolo_preds(
    preds: list, orig_image: Image, model_resolution: Tuple = (640, 640), conf_thres=0.7
) -> np.ndarray:
    photo = orig_image
    preds = non_max_suppression_face(preds[0], conf_thres=conf_thres)[0]
    if preds.shape[0] == 0:
        return preds
    preds = [pred.cpu()[:4] for pred in preds]
    preds = scale_box(
        model_resolution, torch.stack(preds, dim=0), (photo.height, photo.width)
    ).round()
    preds = torch.clamp(preds, min=0)
    preds = preds[preds[:, 0].argsort()]
    return preds.numpy()