import zmq, datetime, time, json, msgpack
from datetime import timedelta
import json
import cv2 as cv
import numpy as np
from scipy.stats import mode
from jetson_inference import poseNet
from jetson_utils import cudaFromNumpy

from camera import RealSenseCamera

net = poseNet("resnet18-body", threshold=0.15)

camera = RealSenseCamera()

context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:40000")


def generate_current_dotnet_datetime_ticks(base_time=datetime.datetime(1, 1, 1)):
    return (
        (datetime.datetime.utcnow() - base_time)
        / datetime.timedelta(microseconds=1)
        * 1e1
    )


def send_payload(pub_sock, topic, message, originatingTime=None):
    payload = {}
    payload["message"] = message
    if originatingTime is None:
        originatingTime = generate_current_dotnet_datetime_ticks()
    payload["originatingTime"] = originatingTime
    pub_sock.send_multipart([topic.encode(), msgpack.dumps(payload)])


colors = np.linspace(0, 255, net.GetNumKeypoints()).astype(np.uint8)
colors = np.squeeze(cv.applyColorMap(colors, cv.COLORMAP_VIRIDIS))
limit = 10

try:
    prev_payload = {'location': '', 'pose': ''}
    history = {}

    hand_raise_count = 0
    while True:
        depth, img = camera.get_frame_stream()
        height = img.shape[0]
        width = img.shape[1]

        frame = cudaFromNumpy(img)

        # get poses from model on rgb frame
        poses = net.Process(frame)

        # process information from poses
        msg = []
        for pose in poses:
            keypoints = pose.Keypoints
            links = pose.Links

            # Bounding box
            x1 = pose.Left
            y1 = pose.Top
            x2 = pose.Right
            y2 = pose.Bottom
            id = pose.ID

            # Keypoint Depth
            kp = keypoints[0]
            depth_mm = int(depth[int(kp.y), int(kp.x)])

            # Where to look
            look = None

            if kp.x < width/3:
                look = 'right'
            elif kp.x > 2 *(width/3):
                look = 'left'
            else: look = 'front'


            # Raise Hand
            kp_dict = {}
            for kp in keypoints:
                kp_ID = kp.ID
                kp_dict[net.GetKeypointName(kp_ID)] = (int(kp.x), int(kp.y))

            r_wrist_kp = kp_dict.get("right_wrist")
            r_elbow_kp = kp_dict.get("right_elbow")

            l_wrist_kp = kp_dict.get("left_wrist")
            l_elbow_kp = kp_dict.get("left_elbow")

            raise_hand = False
            if r_wrist_kp is not None and r_elbow_kp is not None:
                if r_wrist_kp[1] < r_elbow_kp[1]:
                    raise_hand = True

            if l_wrist_kp is not None and l_elbow_kp is not None:
                if l_wrist_kp[1] < l_elbow_kp[1]:
                    raise_hand = True

            # Pose info of that person
            pose_info = {
                "id": id,
                "bbox": [x1, y1, x2, y2],
                "depth": depth_mm,
                "keypoints": kp_dict,
                "links": links,
                "look": look,
                "raise_hand": raise_hand,
            }

            msg.append(pose_info)

            for link in links:
                kp1 = keypoints[link[0]]
                kp2 = keypoints[link[1]]
                kp1_color = (
                    int(colors[kp1.ID][0]),
                    int(colors[kp1.ID][1]),
                    int(colors[kp1.ID][2]),
                )
                kp2_color = (
                    int(colors[kp2.ID][0]),
                    int(colors[kp2.ID][1]),
                    int(colors[kp2.ID][2]),
                )

                # draw link
                cv.line(
                    img,
                    (int(kp1.x), int(kp1.y)),
                    (int(kp2.x), int(kp2.y)),
                    kp1_color,
                    3,
                )

                # draw keypoints

                cv.ellipse(img, (int(kp1.x), int(kp1.y)), (10,10), 0, 0, 360, kp1_color, -1)
                cv.ellipse(img, (int(kp2.x), int(kp2.y)), (10,10), 0, 0, 360, kp2_color, -1)

        for person_id, person in enumerate(msg):
            if person['raise_hand']:
                curr_pose = 'handraise'
            else:
                curr_pose = ''
            
            # person_id = person['id']
        
            if person_id >= len(history):

                history[person_id] = [curr_pose]
            elif len(history[person_id]) < limit:
                history[person_id].append(curr_pose)
            else:
                history[person_id].pop(0)
                history[person_id].append(curr_pose)

        
        pose = ''
        loc = ''
        for person_id, person in enumerate(msg):
            # person_id = person['id']

            if person_id < len(history) :
                pose = mode(history[person_id])[0][0]
                if pose == 'handraise':
                    loc = msg[person_id]['look']
                    break

        payload = {'location': loc, 'pose': pose}
        if pose == 'handraise' and (prev_payload['location'] != payload['location']) and hand_raise_count == 0:
            print('sent payload: {payload}')
            send_payload(socket, 'Remote_PSI_Text', payload)
            hand_raise_count += 1
            time.sleep(0.01)

        if hand_raise_count == 75:
            hand_raise_count = 0
        elif hand_raise_count != 0:
            hand_raise_count += 1

        prev_payload = payload
        cv.putText(img, f"{payload} hand raise counter: {hand_raise_count}", (5, 25), 0, 0.8, (1, 1, 1), 2)
        
        cv.imshow('demo', img)


        # if press esc, quit program
        key = cv.waitKey(1)
        if key == 27:
            break


finally:
    print("Finishing up!")

    camera.stop_stream()
