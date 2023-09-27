import cv2 as cv
import numpy as np
from jetson_inference import poseNet, detectNet
from jetson_utils import cudaFromNumpy

from camera import RealSenseCamera

pose_net = poseNet("resnet18-body", threshold=0.15)
detect_net = detectNet(network="ssd-mobilenet-v2", threshold=0.5)

camera = RealSenseCamera()

colors = np.linspace(0, 255, pose_net.GetNumKeypoints()).astype(np.uint8)
colors = np.squeeze(cv.applyColorMap(colors, cv.COLORMAP_VIRIDIS))
alpha = 0.4

try:
    while True:
        _, img = camera.get_frame_stream()
        height = img.shape[0]
        width = img.shape[1]

        frame = cudaFromNumpy(img)

        poses = pose_net.Process(frame)
        detections = detect_net.Detect(frame, width, height)

        overlay = img.copy()
        for detect in detections:
            ID = detect.ClassID
            y = detect.Top
            x = detect.Left
            y2 = detect.Bottom
            x2 = detect.Right
            item = detect_net.GetClassDesc(ID)
            conf = detect.Confidence

            cv.putText(
                img,
                f"{item}: {conf}",
                (int(x) + 5, int(y) + 25),
                0,
                0.8,
                (255, 255, 255),
                2,
            )
            cv.rectangle(overlay, (int(x), int(y)), (int(x2), int(y2)), ID, -1)

        img = cv.addWeighted(overlay, alpha, img, 1 - alpha, 0)

        for pose in poses:
            keypoints = pose.Keypoints
            links = pose.Links

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
                cv.ellipse(
                    img, (int(kp1.x), int(kp1.y)), (10, 10), 0, 0, 360, kp1_color, -1
                )
                cv.ellipse(
                    img, (int(kp2.x), int(kp2.y)), (10, 10), 0, 0, 360, kp2_color, -1
                )

        cv.namedWindow("pose estimation", cv.WINDOW_AUTOSIZE)
        cv.imshow("pose estimation", img)

        key = cv.waitKey(1)
        if key == 27:
            break
finally:
    print("Finishing up!")
    camera.stop_stream()
