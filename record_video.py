import os
import shutil
import cv2
import time
import argparse

import numpy as np
from datetime import datetime

import sys
required_import_paths = ["~/", "/usr/local/lib/python3.6/pyrealsense2", "~/.local/lib/python3.6/site-packages"]
sys.path = sys.path + required_import_paths

import pyrealsense2 as rs


def main(args):

    # make dirs 
    out_path = os.path.join(args.outdir, args.outfile)

    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    
    depth_dir = os.path.join(out_path, 'depth')
    img_dir = os.path.join(out_path, 'img')
    os.makedirs(depth_dir)
    os.makedirs(img_dir)

    # configure camera pipeline
    pipeline = rs.pipeline()

    config = rs.config()

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)

    profile = pipeline.start(config)

    device = profile.get_device()
    depthSensor = device.first_depth_sensor()
    depthSensor.set_option(rs.option.depth_units, 0.01)

    align_to = rs.stream.color
    align = rs.align(align_to)

    start = time.time()
    frames_saved = 0
    print('Recording')
    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not aligned_depth_frame or not color_frame:
                continue

            depth_img = np.asanyarray(aligned_depth_frame.get_data(), dtype=float)
            color_img = np.asanyarray(color_frame.get_data())

            # save recording frames
            cv2.imwrite(os.path.join(img_dir, f'{frames_saved}.png'), cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
            cv2.imwrite(os.path.join(depth_dir, f'{frames_saved}.png'), depth_img)

            frames_saved += 1

            # if frames_saved == 4500:
            #     print('Stopping recoding at 5 minutes')
            #     break

            # visualize images if visualize flag on 
            if args.visualize:
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=0.03), cv2.COLORMAP_JET)

                images = np.hstack((color_img, depth_colormap))

                cv2.namedWindow('Visualization')
                cv2.imshow('Visualization', images)

                key = cv2.waitKey(1)

                if key & 0xFF == ord('q') or key == 27:
                    cv2.destroyAllWindows()
                    break
    finally:
        print('Stopping Recording')
        end = time.time()

        print(frames_saved/(end-start)) # report frames per second
        pipeline.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', default='./output/', help='Name of output directory, must be created before running')
    parser.add_argument('--outfile', default='video', help='Name of video folder for output')
    parser.add_argument('--visualize', action='store_true', default=False, help='visualize frames while recording, cannot do in headless mode')
    args = parser.parse_args()
    main(args)