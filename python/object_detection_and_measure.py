#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    TensorFlow Lite Object detection and measure height with RealSense.

    Copyright (c) 2021 Nobuo Tsukamoto
    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

import argparse
import colorsys
import os
import random
import time

import cv2
import numpy as np
import pyrealsense2 as rs
import tflite_runtime.interpreter as tflite


WINDOW_NAME = "TensorFlow Lite with Intel RealSense (detection and measure height)"


def draw_rectangle(image, box, color, thickness=3):
    """ Draws a rectangle.
    Args:
        image: The image to draw on.
        box: A list of 4 elements (x1, y1, x2, y2).
        color: Rectangle color.
        thickness: Thickness of lines.
    """
    b = np.array(box).astype(int)
    cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), color, thickness)


def draw_caption(image, box, caption):
    """ Draws a caption above the box in an image.
    Args:
        image: The image to draw on.
        box: A list of 4 elements (x1, y1, x2, y2).
        caption: String containing the text to draw.
    """
    b = np.array(box).astype(int)
    cv2.putText(
        image, caption, (b[0], b[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2
    )
    cv2.putText(
        image, caption, (b[0], b[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1
    )


def set_input_tensor(interpreter, image):
    """ Sets the input tensor.
    Args:
        interpreter: Interpreter object.
        image: a function that takes a (width, height) tuple, 
        and returns an RGB image resized to those dimensions.
    """
    tensor_index = interpreter.get_input_details()[0]["index"]
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image.copy()


def get_output_tensor(interpreter, index):
    """ Returns the output tensor at the given index.
    Args:
        interpreter: Interpreter object.
        index: index
    Returns:
        tensor
    """
    output_details = interpreter.get_output_details()[index]
    tensor = np.squeeze(interpreter.get_tensor(output_details["index"]))
    return tensor


def get_output(interpreter, score_threshold):
    """ Returns list of detected objects.
    Args:
        interpreter: Interpreter object.
        score_threshold: score threshold.
    Returns: bounding_box, class_id, score
    """
    # Get all output details.
    boxes = get_output_tensor(interpreter, 0)
    class_ids = get_output_tensor(interpreter, 1)
    scores = get_output_tensor(interpreter, 2)
    count = int(get_output_tensor(interpreter, 3))

    # Stores results above the threshold in a list.
    results = []
    for i in range(count):
        if scores[i] >= score_threshold:
            result = {
                "bounding_box": boxes[i],
                "class_id": class_ids[i],
                "score": scores[i],
            }
            results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="File path of TF-lite model.", required=True)
    parser.add_argument(
        "--label",
        help="ID of labe. In coco dataformat, ID = 1 is human.",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--threshold", help="threshold to filter results.", default=0.5, type=float
    )
    parser.add_argument("--thread", help="Num threads.", default=1, type=int)
    args = parser.parse_args()

    # Initialize window.
    cv2.namedWindow(
        WINDOW_NAME, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO
    )
    cv2.moveWindow(WINDOW_NAME, 100, 200)

    # Configure depth and color streams
    # For D4XXX
    # W = 848
    # H = 480

    # For L515
    DEPTH_W = 1024
    DEPTH_H = 768
    COLOR_W = 1280
    COLOR_H = 720

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, DEPTH_W, DEPTH_H, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, COLOR_W, COLOR_H, rs.format.bgr8, 30)

    print("[INFO] Starting streaming...")
    pipeline.start(config)
    aligned_stream = rs.align(rs.stream.color)  # alignment between color and depth
    point_cloud = rs.pointcloud()
    print("[INFO] Camera ready.")

    # Initialize TF-Lite interpreter.
    interpreter = tflite.Interpreter(model_path=args.model, num_threads=args.thread)
    interpreter.allocate_tensors()
    _, input_height, input_width, _ = interpreter.get_input_details()[0]["shape"]

    start_ms = time.perf_counter()

    while True:
        frames = pipeline.wait_for_frames()
        frames = aligned_stream.process(frames)
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        points = point_cloud.calculate(depth_frame)
        verts = (
            np.asanyarray(points.get_vertices())
            .view(np.float32)
            .reshape(-1, COLOR_W, 3)
        )  # xyz

        # Convert images to numpy arrays.
        color_image = np.asanyarray(color_frame.get_data())
        scaled_size = (int(COLOR_W), int(COLOR_H))
        resize_image = cv2.resize(color_image, (input_width, input_height))
        resize_image = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB)

        set_input_tensor(interpreter, resize_image)
        interpreter.invoke()
        objs = get_output(interpreter, args.threshold)

        # Display result.
        for obj in objs:
            # Convert the bounding box figures from relative coordinates
            # to absolute coordinates based on the original resolution.
            ymin, xmin, ymax, xmax = obj["bounding_box"]
            xmin = int(xmin * scaled_size[0])
            xmax = int(xmax * scaled_size[0])
            ymin = int(ymin * scaled_size[1])
            ymax = int(ymax * scaled_size[1])

            # Draw a rectangle
            draw_rectangle(color_image, (xmin, ymin, xmax, ymax), (255, 209, 0))

            # x,y,z of bounding box
            bbox = (xmin, ymin, xmax - xmin, ymax - ymin)
            obj_points = verts[
                int(bbox[1]) : int(bbox[1] + bbox[3]),
                int(bbox[0]) : int(bbox[0] + bbox[2]),
            ].reshape(-1, 3)
            zs = obj_points[:, 2]
            ys = obj_points[:, 1]

            z = np.median(zs)
            ys = np.delete(
                ys, np.where((zs < z - 1) | (zs > z + 1))
            )  # take only y for close z to prevent including background

            my = np.amin(ys, initial=1)
            My = np.amax(ys, initial=-1)

            height = My - my  # add next to rectangle print of height using cv library
            height = float("{:.2f}".format(height))
            height_txt = str(height) + "[m]"
            draw_caption(color_image, (xmin, ymin, xmax, ymax), height_txt)

        end_ms = time.perf_counter()
        elapsed_ms = (end_ms - start_ms) * 1000
        fps = (1.0 / elapsed_ms) * 1000
        start_ms = end_ms

        fps_text = "{0:.2f} ms, ".format(elapsed_ms) + "{0:.2f} FPS".format(fps)
        draw_caption(color_image, (10, 30), fps_text)

        # Display
        cv2.imshow(WINDOW_NAME, color_image)
        key = cv2.waitKey(10) & 0xFF
        if key == ord("q"):
            break

    # Stop streaming
    pipeline.stop()


if __name__ == "__main__":
    main()
