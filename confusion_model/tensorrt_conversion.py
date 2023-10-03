from facenet_pytorch import InceptionResnetV1
import os
import torchvision.transforms as transforms
from typing import Tuple
import torch 
from torch2trt import torch2trt, TRTModule
import PIL.Image as Img
from PIL.Image import Image
import time 


def convert_model(model, data, dtype: str = "f16", device: str = "cuda"):
    with torch.no_grad(): 
     
        model = model.eval()
        model_fp_mode=False
        if dtype == "f16": 
            model_fp_mode = True
            data = data.half()
        else: 
            data = data.float()
        if model_fp_mode: 
            model_trt = torch2trt(model.half(), [data], fp16_mode=True)
        else: 
            print("USING FULL PRECISION")
            model_trt = torch2trt(model, [data], fp16_mode=False)
    return model, model_trt

def save_trt_model(trt_model, save_filepath):
    dir = "/".join(save_filepath.split("/")[:-1])
    if not os.path.isdir(dir): 
        os.makedirs(dir, exists_ok=True)
    torch.save(trt_model.state_dict(), save_filepath)

def load_model_path(model_path): 
    model_trt = TRTModule()
    model_trt.load_state_dict(torch.load(model_path))
    return model_trt

if __name__ == "__main__":
    file_path = "/usr0/home/nvaikunt/full_images/image_0.png"
    photo = Img.open(file_path)
    transform = transforms.Compose([
        transforms.PILToTensor()])
    data = transform(photo).unsqueeze(0)
    data = data.to("cuda")
    print(data.dtype)
    print(data.shape)
    model = InceptionResnetV1(device = "cuda")
    resnet, resnet_trt = convert_model(model, data, dtype="f16")

    # Save TRT Model
    #save_trt_model(resnet_trt, "./data/FCN_CNN_512_3.pth")
    #trt_model = load_model_path("./data/FCN_CNN_512_3.pth")
    trt_model = resnet_trt
    with torch.no_grad(): 
        for i in range(1, 35):
            new_photo = Img.open(f"/usr0/home/nvaikunt/full_images/image_{i}.png")
            new_data = transform(new_photo).unsqueeze(0)
            new_data = new_data.to("cuda")

            start = time.time()
            full_model = InceptionResnetV1(device = "cuda").eval().half()
            # input_full = new_data.float()
            input_half = new_data.half()
            feats = full_model(input_half)
            print(f"Time to execute regular featurizer {time.time() - start}")
            print(feats.flatten()[0:10])

            start = time.time()
            # input_half = new_data.half()
            trt_feats = trt_model(input_half)
            # trt_feats = trt_model(input_full)
            print(f"Time to execute TRT featurizer  {time.time() - start}")
            print(trt_feats.flatten()[0:10])
            print('max error: %f' % float(torch.max(torch.abs(feats - trt_feats))))

   
