"""
Done: Accept images (thread), get predictions (thread). Env /usr0/home/sohamdit/Jetson/jetson_venv
"""
import cv2 as cv
import threading
import time
from collections import deque
import numpy as np
from config import *
from zmq_utils import *
from confusion_model.inference import ConfusionInference
from confusion_model.constants import *
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
    inference_model = ConfusionInference(
        load_model_path="/usr0/home/nvaikunt/FCN_CNN_512_3.bin",
        data_type="window",
        multiclass=False,
        label_dict=EMOTION_NO,
        device="cuda",
        haar_path=None,
        extractor="stable",
        tensor_rt = True
    )
    window_len = inference_model.window_len
    num_preds = 0
    start = time.time()

    while buffer:
        if len(buffer) > window_len:
            with buffer_lock:
                buffer_outputs = [buffer.popleft() for _ in range(window_len)]

            current_images = []
            for img, _ in buffer_outputs:
                h, w = img.size
                print("hello", h, w)
                curr_image = img.resize((3 * h // 4,  3 * w // 4))
                print(curr_image.size)
                current_images.append(curr_image)
            
            preds = inference_model.run_inference(current_images)
            ot = send_payload(pub_socket_to_psi, "cv-preds", np.array(preds.reshape(-1,).tolist()).tobytes())
            print(f"hello there: {ot}", preds.reshape(-1,).shape, preds.reshape(-1,).tolist())
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
    print(f"Total number of predictions {num_preds}")
    print(f"Total inference time: {time.time() - start}")

def read_frames():
    try:
        while True:
            frame, originatingTime = readFrame(sub_socket_to_psi)
            img = base64.b64decode(frame)
            npimg = np.fromstring(img, dtype=np.uint8)
            img = Image.fromarray(cv2.imdecode(npimg, 1))
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
