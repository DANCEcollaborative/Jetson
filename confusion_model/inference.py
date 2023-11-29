import argparse
from argparse import Namespace
from typing import List, Tuple, Optional
import os
import datetime
import csv

import PIL.Image as PIL_Image
from PIL.Image import Image
from PIL import ImageDraw
import torch
from torchvision.models.detection import (
    keypointrcnn_resnet50_fpn,
    KeypointRCNN_ResNet50_FPN_Weights,
)
from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights,
    ssdlite320_mobilenet_v3_large,
)
from torchvision.transforms import v2

from yolo_models.yolo import Model
from .utils import (
    generate_face_bbox,
    generate_keypoints,
    generate_person_bboxes,
    load_model, 
    get_cropped_image
)



class ConfusionDetectionInference:
    def __init__(
        self,
        model_save_path: str,
        model_config: Namespace,
        device: str,
        threshold_dict: dict = None,
        yolo_config: str = None,
        yolo_model_pth: str = None,
        save_pred: bool = False
    ) -> None:
        # Load Model Class
        self.model = load_model(model_args=model_config)
        self.model_config = model_config
        if self.model_config.model_type not in ['keypoints', 'full_image', 'single_body_image', 'single_face_image']:
            raise ValueError(f"Must pass in valid model type in {['keypoints', 'full_image', 'single_body_image', 'single_face_image']}")
        self.device = device
        self.model.to(self.device)

        # Load Model Weight
        self.model.load_state_dict(
            torch.load(model_save_path, map_location=self.device)
        )
        self.model.eval()
        # Relevant Image Transforms
        self.tensor_transforms = v2.Compose(
            [v2.PILToTensor(), v2.ConvertDtype()]
        )
        # Save Feature Thresholds
        if threshold_dict is None:
            self.thresholds = {"person": 0.8, "keypoints": 0.9, "face": 0.7}
        else:
            assert (
                "person" in threshold_dict
                and "keypoints" in threshold_dict
                and "face" in threshold_dict
            ), "Threshold Dict must provide threshold for person, keypoint, and face"
            self.thresholds = threshold_dict

        # Load Person BBox:
        self.person_detector = ssdlite320_mobilenet_v3_large(
            weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        )
        self.person_detector.eval()
        self.person_detector.to(self.device)

        if self.model_config.model_type != "single_body_image":
            if yolo_config is None or yolo_model_pth is None:
                raise TypeError(
                    "None type was provided for yolo model config "
                    "and/or yolo model path. Please use yolo model path"
                )
            # Intiailize Face Featurizer
            self.face_detector = Model(cfg=yolo_config)
            self.face_detector.load_state_dict(
                torch.load(yolo_model_pth, map_location=torch.device("cpu"))
            )
            self.face_detector.eval()
            self.face_detector.to(self.device)

        if self.model_config.model_type == "keypoints":
            # Intialize Keypoint Featurizer
            self.keypoint_model = keypointrcnn_resnet50_fpn(
                weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
            )
            self.keypoint_model.eval()
            self.keypoint_model.to(self.device)
        self.save_pred = save_pred
        self.csv_writer = None 
        self.all_preds = []
        if self.save_pred: 
            os.makedirs(f"./saved_preds/annotations", exist_ok=True)
            os.makedirs(f"./saved_preds/images/{datetime.today().strftime('%Y-%m-%d')}", exist_ok=True)
            f = open(f"./saved_preds/annotations/preds_{datetime.today().strftime('%Y-%m-%d')}.csv", "a", newline='')
            self.csv_writer = csv.writer(f)
            self.csv_writer.writerow(["ID", "Time", "Valence", "Arousal", "Dominance"])

    def get_list_of_person_bboxes(self, full_image: Image) -> List[Image]:
   
        people_bboxes = generate_person_bboxes(
            self.person_detector,
            full_image,
            self.tensor_transforms,
            self.device,
            self.thresholds["person"],
        )
        if not people_bboxes:
            return None
        people_images = [
            get_cropped_image(full_image, person_bbox) for person_bbox in people_bboxes
        ]
        return people_images

    def get_full_img_tensor(self, image: Image):
        image = image.resize((160, 160))
        return self.tensor_transforms(image)

    def get_face_tensor_from_img(self, full_image: Image) -> Optional[torch.Tensor]:
        face_bbox = generate_face_bbox(
            self.face_detector,
            full_image,
            self.tensor_transforms,
            self.device,
            self.thresholds["face"],
        )
        if len(face_bbox) == 0:
            return None
        face_image = get_cropped_image(full_image, face_bbox)
        face_image = face_image.resize((160, 160))
        return self.tensor_transforms(face_image)

    def get_relevant_feats(
        self, person_img: Image
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        body_feat, face_feat = None, None
        if self.model_config.model_type == "single_body_image":
            body_feat = self.get_full_img_tensor(person_img)
        elif self.model_config.model_type == "single_face_image":
            face_feat  = self.get_face_tensor_from_img(person_img)
        elif self.model_config.model_type == "keypoints":
            face_feat  = self.get_face_tensor_from_img(person_img)
            body_feat = generate_keypoints(
                self.keypoint_model,
                person_img,
                self.tensor_transforms,
                device=self.device,
                detect_threshold=self.thresholds["keypoints"],
                inference=True,
            )
            if len(body_feat) == 0:
                body_feat = None
            else:
                body_feat = body_feat[0]
        else:
            face_feat = self.get_face_tensor_from_img(person_img)
            body_feat  = self.get_full_img_tensor(person_img)
        return body_feat, face_feat

    def postprocess(self, pred_list: List[list]) -> bool:
        confusion = False
        valence_threshold = 5
        arousal_threshold = 4
        dominance_threshold = 6
        for pred in pred_list:
            pred_valence, pred_arousal, pred_dominance = pred
            if (
                pred_valence < valence_threshold
                and pred_arousal < arousal_threshold
                and pred_dominance < dominance_threshold
            ):
                confusion = True
        return confusion

    def run_inference(self, image: Image) -> bool:
        person_images = self.get_list_of_person_bboxes(image)
        id = 0
        if person_images is None:
            return [10.0, 10.0, 10.0]
        pred_list = []
        for person in person_images:
            body_feat, face_feat = self.get_relevant_feats(person)
            # Get outputs
            if self.model_config.model_type == "single_body_image":
                if body_feat is None:
                    continue
                body_feat = body_feat.to(self.device).unsqueeze(0)
                outputs = self.model(body_feat)
            elif self.model_config.model_type == "single_face_image":
                if face_feat is None:
                    continue
                face_feat = face_feat.to(self.device).unsqueeze(0)
                outputs = self.model(face_feat)
            else:
                if body_feat is None or face_feat is None:
                    continue
                body_feat = body_feat.to(self.device).unsqueeze(0)
                face_feat = face_feat.to(self.device).unsqueeze(0)
                outputs = self.model(body_feat, face_feat)

            preds = torch.clamp(outputs, 1, 10).squeeze(0).detach().cpu().tolist()
            if self.save_pred: 
                timestamp = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
                saved_outputs = [timestamp]
                saved_outputs.extend(preds)
                self.all_preds.append(saved_outputs)
                person_draw = ImageDraw.Draw(person)
                person_draw.text((5, 5), timestamp, fill=(255, 0, 0))        
                person_draw.text((5, 25), preds, fill=(255, 0, 0))      
                person.save(f"./saved_preds/images/{datetime.today().strftime('%Y-%m-%d')}/pred_{id}.jpg")     
                id += 1


            pred_list.append(preds)
        print(pred_list)

        if not pred_list:
            return 0
        else:
            return self.postprocess(pred_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EMOTIC Model Inference")
    yolo_config_full = "./models/yolov5n.yaml"
    yolo_weights = "./model_weights/pretrained/yolov5n-face_new.pt"
    # Model Args

    parser.add_argument("--model_save_path", type=str, default="")
    parser.add_argument("--model_type", type=str, default="keypoint")
    parser.add_argument("--num_fusion_layers", type=int, default=3)
    parser.add_argument("--hidden_sz_const", type=int, default=512)
    parser.add_argument("--post_concat_feat_sz", type=int, default=512)
    parser.add_argument("--hidden_size_strategy", type=str, default="constant")

    # Other Args
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    

    # Can hardcode this if needed
    model_args = Namespace(
        model_type=args.model_type,
        num_fusion_layers=args.num_fusion_layers,
        hidden_sz_const=args.hidden_sz_const,
        post_concat_feat_sz=args.post_concat_feat_sz,
        hidden_sz_strat=args.hidden_size_strategy,
        dropout=0.0
    )

    confusion_detector = ConfusionDetectionInference(
        model_save_path=args.model_save_path,
        model_config=model_args,
        device=args.device,
        yolo_config = "./models/yolov5n.yaml",
        yolo_model_pth= "./model_weights/pretrained/yolov5n-face_new.pt"
    )
    
    demo_files = ["./emotic/emodb_small/images/68ywy6tv3qzjmyl3ip.jpg",  "./emotic/emodb_small/images/cbjt98z7oq43vtftdu.jpg"]
    for file in demo_files: 
        demo_img = PIL_Image.open(file)
        confusion_ind = confusion_detector.run_inference(demo_img)
        print(confusion_ind)


