import torch
import sys
import pickle
import PIL.Image as Img
from PIL.Image import Image
from PIL import ImageDraw
import torchvision.transforms as transforms
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from yolo_models.yolo import Model
from yolo_utils.general import non_max_suppression_face

if __name__ == "__main__":
    new_photo = Img.open(f"/usr0/home/nvaikunt/full_images/image_0.png") 
    new_photo = new_photo.resize((640, 640))
    draw = ImageDraw.Draw(new_photo)
    transform = transforms.Compose([transforms.PILToTensor(), transforms.ConvertImageDtype(torch.float32)])
    new_data = transform(new_photo) 
    model_path = "/usr0/home/nvaikunt/Jetson/data/yolov5n-face_new.pt"
    cfg_dict_pth = "/usr0/home/nvaikunt/Jetson/data/yolov5n-cfg_dict.pkl"

    with open(cfg_dict_pth, 'rb') as handle:
        cfg_dict = pickle.load(handle)
    model = Model(cfg=cfg_dict)
    model.load_state_dict(torch.load(model_path))
    with torch.no_grad():
        new_data = new_data.to("cuda").unsqueeze(0)
        model = model.to("cuda").eval()
        preds = model(new_data)
        preds = non_max_suppression_face(preds[0])[0]
        preds = [pred.cpu()[:4] for pred in preds]
        for pred in preds: 
            x1, y1, x2, y2 = pred
            draw.rectangle([(x1,x2),(y1,y2)],outline="green")
        new_photo.show()
