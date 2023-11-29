"""
Done: Accept images (thread), get predictions (thread). Env /usr0/home/sohamdit/Jetson/jetson_venv
"""
import cv2 as cv
import threading
import time
from argparse import Namespace
from collections import deque
import numpy as np
from config import *
from zmq_utils import *
from confusion_model.inference import ConfusionDetectionInference
from legacy_confusion_model.constants import *
from PIL import Image
import base64
import cv2


# context = zmq.Context()
# socket = context.socket(zmq.SUB)
# socket.bind('tcp://*:60004')
sub_socket_to_psi = create_sub_socket('tcp://localhost:40004')
sub_socket_to_psi.setsockopt_string(zmq.SUBSCRIBE, "images-psi-to-python")

pub_socket_to_psi = create_socket(ip_address=f"tcp://*:{confusion_classifier_res_port}")

# Initialize global buffer
BUF_MAX_LEN = 6
buffer = deque(maxlen=BUF_MAX_LEN) 
buffer_lock = threading.Lock()

def confusion_cnn_embed():
    model_args = Namespace(
        model_type="keypoints",
        num_fusion_layers=3,
        hidden_sz_const=512,
        post_concat_feat_sz=512,
        hidden_sz_strat="constant",
        dropout=0.0
    )

    inference_model = ConfusionDetectionInference(
        model_save_path="/usr0/home/nvaikunt/Jetson/model_weights/keypoints.pt",
        model_config=model_args,
        device="cuda",
        yolo_config = "./yolo_models/yolov5n.yaml",
        yolo_model_pth= "/usr0/home/nvaikunt/Jetson/data/yolov5n-face_new.pt",
        save_pred=True
    )
    
    num_preds = 0
    start = time.time()

    while buffer:

        current_image, _ = buffer.popleft()
        pred = inference_model.run_inference(current_image)
        ot = send_payload(pub_socket_to_psi, "cv-preds", np.array(pred).tobytes())
        print(f"hello there: {ot}",  pred)
        num_preds += 1
            # feat_tensor = inference_model.extract_cnn_feats(curr_image)
            # if feat_tensor is not None:
            #     inference_model.feats.append(feat_tensor)
            # if len(inference_model.feats) == window_len:
            #     preds = inference_model.run_inference()
            #     payload = {"timestamp": timestamp, "preds": preds}  #ToDo: Convert "preds" type to something that send_payload expects
            #     send_payload(socket, "Remote_PSI_Text", payload)
            #     print(preds)
            #     inference_model.feats.pop(0)
            # send_payload(socket, "cvpreds", payload, originatingTime=buffer_outputs[0][1]) # sending the time when the first image of input window was captured as the originatingTime
        time.sleep(0.01)
    inference_model.close()
    print(f"Total number of predictions {num_preds}")
    print(f"Total inference time: {time.time() - start}")

def read_frames():
    try:
        while True:
            frame, originatingTime = readFrame(sub_socket_to_psi)
            img = base64.b64decode(frame)
            npimg = np.fromstring(img, dtype=np.uint8)
            img = Image.fromarray(cv2.cvtColor(cv2.imdecode(npimg, 1), cv2.COLOR_BGR2RGB))
            # if frame_count % 10 == 0:  # Add every 10th frame to buffer
            with buffer_lock:
                buffer.append((img, originatingTime))  # Appending image and current time as tuple to buffer

            # time.sleep(0.4)

    finally:
        print("Finishing up!")


def main():
    capture_thread = threading.Thread(target=read_frames, daemon=True)
    inference_thread = threading.Thread(target=confusion_cnn_embed, daemon=True)

    capture_thread.start()
    inference_thread.start()

    capture_thread.join()
    inference_thread.join()


if __name__ == "__main__":
    main()
