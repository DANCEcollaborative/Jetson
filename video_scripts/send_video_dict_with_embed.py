import cv2 as cv
from video_scripts.camera import RealSenseCamera
import threading
import time
from collections import deque
import sys
from confusion_model.inference import ConfusionInference
from config import *
from zmq_utils import *

camera = RealSenseCamera()

context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind(f"tcp://*:{confusion_classifier_res_port}")

# Initialize global buffer
BUF_MAX_LEN = 6
buffer = deque(maxlen=BUF_MAX_LEN) 
buffer_lock = threading.Lock()

def confusion_cnn_embed():
    inference_model = ConfusionInference(
        load_model_path="/home/teledia/Desktop/nvaikunt/ConfusionDataset/data/FCN_CNN_512_3.bin",
        data_type="window",
        multiclass=False,
        label_dict=EMOTION_NO,
        device="cuda",
        haar_path="/home/teledia/Desktop/nvaikunt/ConfusionDataset/data/haarcascade_frontalface_alt_cuda.xml"
    )
    window_len = inference_model.window_len
    num_preds = 0
    start = time()

    while buffer:
        if len(buffer) > window_len:
            with buffer_lock:
                buffer_outputs = [buffer.popleft() for _ in range(window_len)]

            current_images = []
            for bo in buffer_outputs:
                h, w = bo[0].size
                print("hello", h, w)
                curr_image = bo[0].resize((3 * h // 4,  3 * w // 4))
                print(curr_image.size)
                curr_images.append(curr_image)
            
            preds = inference_model.run_inference()
            payload = preds #ToDo: Convert "preds" type to something that send_payload expects
            print(preds)
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
            send_payload(socket, "cvpreds", payload, originatingTime=buffer_outputs[0][1]) # sending the time when the first image of input window was captured as the originatingTime
        time.sleep(0.01)
    print(f"Total number of predictions {num_preds}")
    print(f"Total inference time: {time() - start}")

def capture_frames():
    try:
        history = {}
        while True:
            depth, img = camera.get_frame_stream()
            height = img.shape[0]
            width = img.shape[1]
            print(height, width)

            # if frame_count % 10 == 0:  # Add every 10th frame to buffer
            with buffer_lock:
                buffer.append((img, generate_current_dotnet_datetime_ticks()))  # Appending image and current time as tuple to buffer

            time.sleep(0.01)

            cv.imshow("demo", img)

            # print('msg:', msg)
            # print('msg length', len(msg))

            key = cv.waitKey(1)
            if key == 27:
                break

            # time.sleep(0.4)

    finally:
        print("Finishing up!")

        camera.stop_stream()

def main():
    capture_thread = threading.Thread(target=capture_frames, daemon=True)
    # inference_thread = threading.Thread(target=perform_inference, daemon=True)

    capture_thread.start()
    # inference_thread.start()

    capture_thread.join()
    # inference_thread.join()


if __name__ == "__main__":
    main()