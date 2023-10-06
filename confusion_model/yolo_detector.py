import torch
import sys
import pickle
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from yolo_models.yolo import Model

if __name__ == "__main__":
    model_path = "/usr0/home/nvaikunt/Jetson/data/yolov5n-face_new.pt"
    cfg_dict_pth = "/usr0/home/nvaikunt/Jetson/data/yolov5n-cfg_dict.pkl"

    with open(cfg_dict_pth, 'rb') as handle:
        cfg_dict = pickle.load(handle)
    model = Model(cfg=cfg_dict)
    model.load_state_dict(torch.load(model_path))