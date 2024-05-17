import logging

import matplotlib.pyplot as plt
import numpy as np
import pyrealsense2 as rs

logger = logging.getLogger(__name__)


class RealSenseCamera:
    def __init__(self,
                 device_id,
                 width=640,
                 height=480,
                 fps=6):
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps

        self.pipeline = None
        self.scale = None
        self.intrinsics = None

        self.aligned_depth_frame = None

    def connect(self):
        # Start and configure
        self.pipeline = rs.pipeline()
        config = rs.config()
        # import pdb;pdb.set_trace()
        # config.enable_device(str(self.device_id))
        # config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, 30)#self.fps)
        # config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 1920, 1080, rs.format.rgb8, 30)
        cfg = self.pipeline.start(config)

        # Determine intrinsics
        rgb_profile = cfg.get_stream(rs.stream.color)
        self.intrinsics = rgb_profile.as_video_stream_profile().get_intrinsics()

        # Determine depth scale
        self.scale = cfg.get_device().first_depth_sensor().get_depth_scale()

    def get_image_bundle(self):
        frames = self.pipeline.wait_for_frames()

        align = rs.align(rs.stream.color)
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.first(rs.stream.color)
        self.aligned_depth_frame = aligned_frames.get_depth_frame()

        depth_image = np.asarray(self.aligned_depth_frame.get_data(), dtype=np.float32)
        
        depth_image_s = depth_image * self.scale
        color_image = np.asanyarray(color_frame.get_data())

        depth_image_s = np.expand_dims(depth_image_s, axis=2)

        return {
            'rgb': color_image,
            'aligned_depth': depth_image_s,
            'unscale_depth': depth_image,
        }

    def plot_image_bundle(self):
        images = self.get_image_bundle()

        rgb = images['rgb']
        depth = images['aligned_depth']

        fig, ax = plt.subplots(1, 2, squeeze=False)
        ax[0, 0].imshow(rgb)
        m, s = np.nanmean(depth), np.nanstd(depth)
        ax[0, 1].imshow(depth.squeeze(axis=2), vmin=m - s, vmax=m + s, cmap=plt.cm.gray)
        ax[0, 0].set_title('rgb')
        ax[0, 1].set_title('aligned_depth')

        plt.show()

    def get_dist(self, u, v):
        return self.aligned_depth_frame.get_distance(u, v)


if __name__ == '__main__':
    cam = RealSenseCamera(device_id=830112070066)
    cam.connect()
    while True:
        cam.plot_image_bundle()
