#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    RealSense PCD Viewer with Open3D.

    Copyright (c) 2021 Nobuo Tsukamoto
    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

import argparse
import json
import os

import numpy as np
import open3d as o3d


class ViewerWithCallback:
    def __init__(self, config, device):
        self.flag_exit = False

        if config is not None and os.path.isfile(config):
            print("Load json: ", config)
            with open(config) as f:
                rs_config = o3d.t.io.RealSenseSensorConfig(json.load(f))
        else:
            print("Use default config")
            rs_config = o3d.t.io.RealSenseSensorConfig()

        # Initialize RealSense.
        self.sensor = o3d.t.io.RealSenseSensor()
        self.sensor.init_sensor(sensor_config=rs_config, sensor_index=device)

        # Get camera intrinsic
        self.metadata = self.sensor.get_metadata()
        self.intrinsics = self.metadata.intrinsics

        # We will not display the background of objects more than
        #  clipping_distance_in_meters meters away
        self.clipping_distance_in_meters = 3  # 3 meter
        depth_scale = 1.0 / self.metadata.depth_scale
        self.clipping_distance = self.clipping_distance_in_meters / depth_scale

    def escape_callback(self, vis):
        print("Escape callback")
        self.flag_exit = True
        return False

    def run(self):
        glfw_key_escape = 256
        pcd = o3d.geometry.PointCloud()
        flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]

        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.register_key_callback(glfw_key_escape, self.escape_callback)
        vis.create_window("Open3d Realsense pointcloud viewer")
        print("Sensor initialized. Press [ESC] to exit.")

        self.sensor.start_capture()
        vis_geometry_added = False
        while not self.flag_exit:
            # Note: In the case of PointCloud, it is necessary to align with the depth.
            rgbd = self.sensor.capture_frame(True, True)
            if rgbd is None:
                continue

            temp_rgbd = rgbd.to_legacy_rgbd_image()
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                temp_rgbd.color,
                temp_rgbd.depth,
                depth_scale=self.metadata.depth_scale,
                depth_trunc=self.clipping_distance_in_meters,
                convert_rgb_to_intensity=False,
            )
            temp_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_image, self.intrinsics
            )
            temp_pcd.transform(flip_transform)
            pcd.points = temp_pcd.points
            pcd.colors = temp_pcd.colors

            if not vis_geometry_added:
                vis.add_geometry(pcd)
                vis_geometry_added = True

            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()

        self.sensor.stop_capture()
        vis.destroy_window()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RealSene PointCloud Viewer.")
    parser.add_argument(
        "--config", type=str, required=True, help="Input json for realsense config"
    )
    parser.add_argument(
        "--device", type=int, default=0, help="input realsense device id"
    )
    args = parser.parse_args()

    # Display device list
    o3d.t.io.RealSenseSensor.list_devices()

    # Set device id
    device = args.device
    if device < 0 or device > 255:
        print("Unsupported device id, fall back to 0")
        device = 0

    # Run
    v = ViewerWithCallback(args.config, device)
    v.run()
