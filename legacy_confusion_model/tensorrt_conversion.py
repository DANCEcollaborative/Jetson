from facenet_pytorch import InceptionResnetV1
import pickle
import os
import sys
from pathlib import Path
import torchvision.transforms as transforms
from typing import Tuple
import torch 
import torch.nn as nn
from torch2trt import torch2trt, TRTModule
import PIL.Image as Img
from PIL.Image import Image
import time 
sys.path.append(str(Path(__file__).resolve().parent.parent))
from yolo_models.yolo import Model
import yolo_models
from yolo_utils.activations import Hardswish, SiLU

def convert_model(model, data_shape: tuple, dtype: str = "f16", device: str = "cuda"):
    with torch.no_grad(): 
        data = torch.rand(data_shape).to(device)
        model = model.eval()
        model_fp_mode=False
        if dtype == "f16": 
            model_fp_mode = True
            data = data.half()
            model = model.half()
        else: 
            data = data.float()
        if model_fp_mode: 
            model_trt = torch2trt(model, [data], fp16_mode=True)
        else: 
            print("USING FULL PRECISION")
            model_trt = torch2trt(model, [data], fp16_mode=False)
    return model, model_trt


def save_trt_model(trt_model, save_filepath):
    dir = "/".join(save_filepath.split("/")[:-1])
    if not os.path.isdir(dir): 
        os.makedirs(dir, exists_ok=True)
    torch.save(trt_model.state_dict(), save_filepath)

def load_model(model_path): 
    model_trt = TRTModule()
    model_trt.load_state_dict(torch.load(model_path))
    return model_trt

if __name__ == "__main__":

    data_shape = (1, 3, 640, 640)
    # model = InceptionResnetV1(device = "cuda", pretrained='vggface2')
    # resnet, resnet_trt = convert_model(model, data_shape, dtype="f16")
    model_path = "/usr0/home/nvaikunt/Jetson/data/yolov5n-face_new.pt"
    cfg_dict_pth = "/usr0/home/nvaikunt/Jetson/data/yolov5n-cfg_dict.pkl"

    with open(cfg_dict_pth, 'rb') as handle:
        cfg_dict = pickle.load(handle)
    model = Model(cfg=cfg_dict)
    model.load_state_dict(torch.load(model_path))
    yolo, yolo_trt = convert_model(model.to("cuda"), data_shape=data_shape, dtype="f32")
    # Save TRT Model
    save_trt_model(yolo_trt, "./data/YOLO_FACE_Nano.pth")
    trt_model = load_model("./data/YOLO_FACE_Nano.pth")
    # full_model = InceptionResnetV1(device = "cuda", pretrained='vggface2').eval()
    """
    with torch.no_grad(): 
        for i in range(1, 33):
            new_photo = Img.open(f"/usr0/home/nvaikunt/full_images/image_{i}.png") 
            transform = transforms.Compose([transforms.PILToTensor()])
            new_data = transform(new_photo) 
            new_data = new_data.to("cuda").unsqueeze(0)

            
            input_full = new_data.float()
            # input_half = new_data.half()
            start = time.time()
            feats = full_model(input_full)
            print(f"Time to execute regular featurizer {(time.time() - start) * 1000} ms")
            print(feats.flatten()[0:10])

            start = time.time()
            input_half = new_data.half()
            trt_feats = trt_model(input_half)
            # trt_feats = trt_model(input_full)
            print(f"Time to execute TRT featurizer {(time.time() - start) * 1000} ms")
            print(trt_feats.flatten()[0:10])
            print('max error: %f' % float(torch.max(torch.abs(feats - trt_feats)))) """

   
