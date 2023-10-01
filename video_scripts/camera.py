from video_scripts.video_config import *
import pyrealsense2 as rs
import numpy as np


class RealSenseCamera:
    def __init__(self, res=(1280, 720)):
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.res = res

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = self.config.resolve(pipeline_wrapper)

        device = pipeline_profile.get_device()
        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == "RGB Camera":
                found_rgb = True
                break
        if not found_rgb:
            raise Exception(
                "This camera configuration requires Depth camera with Color sensor"
            )

        # Enable Depth and Color streams from RealSense Camera
        # self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        # self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        self.config.enable_stream(
            rs.stream.color, self.res[0], self.res[1], rs.format.bgr8, 30
        )
        self.config.enable_stream(
            rs.stream.depth, self.res[0], self.res[1], rs.format.z16, 30
        )

        # Start streaming
        self.pipeline.start(self.config)
        align_to = rs.stream.color
        self.align = rs.align(align_to)

    def get_frame_stream(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            # If there is no frame, probably camera not connected, return False
            raise Exception(
                "Could not retrieve depth and/or color frame. Check that camera is connected."
            )

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        return depth_image, color_image

    def stop_stream(self):
        # Stop streaming
        self.pipeline.stop()
