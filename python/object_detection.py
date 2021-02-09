#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    TensorFlow Lite Object detection with RealSense.

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


WINDOW_NAME = "TensorFlow Lite with Intel RealSense (detection)"


def read_label_file(file_path):
    """ Function to read labels from text files.
    Args:
        file_path: File path to labels.
    Returns:
        list of labels
    """
    with open(file_path, "r") as f:
        lines = f.readlines()

    ret = {}
    for line in lines:
        pair = line.strip().split(maxsplit=1)
        ret[int(pair[0])] = pair[1].strip()
    return ret


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


def random_colors(N):
    """ Random color generator.
    """
    N = N + 1
    hsv = [(i / N, 1.0, 1.0) for i in range(N)]
    colors = list(
        map(lambda c: tuple(int(i * 255) for i in colorsys.hsv_to_rgb(*c)), hsv)
    )
    random.shuffle(colors)
    return colors


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
    parser.add_argument("--label", help="File path of label file.", required=True)
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
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    print("[INFO] Starting streaming...")
    pipeline.start(config)
    print("[INFO] Camera ready.")

    # Initialize TF-Lite interpreter.
    interpreter = tflite.Interpreter(model_path=args.model, num_threads=args.thread)
    interpreter.allocate_tensors()
    _, input_height, input_width, _ = interpreter.get_input_details()[0]["shape"]

    # Read label and generate random colors.
    labels = read_label_file(args.label) if args.label else None
    last_key = sorted(labels.keys())[len(labels.keys()) - 1]
    random.seed(42)
    colors = random_colors(last_key)

    start_ms = time.perf_counter()

    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        # Convert images to numpy arrays.
        color_image = np.asanyarray(color_frame.get_data())
        scaled_size = (color_frame.width, color_frame.height)
        resize_image = cv2.resize(color_image, (input_width, input_height))
        resize_image = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB)

        set_input_tensor(interpreter, resize_image)
        interpreter.invoke()
        objs = get_output(interpreter, args.threshold)

        # Display result.
        for obj in objs:
            class_id = int(obj["class_id"])
            caption = "{0}({1:.2f})".format(labels[class_id], obj["score"])

            # Convert the bounding box figures from relative coordinates
            # to absolute coordinates based on the original resolution.
            ymin, xmin, ymax, xmax = obj["bounding_box"]
            xmin = int(xmin * scaled_size[0])
            xmax = int(xmax * scaled_size[0])
            ymin = int(ymin * scaled_size[1])
            ymax = int(ymax * scaled_size[1])

            # Draw a rectangle and caption.
            draw_rectangle(color_image, (xmin, ymin, xmax, ymax), colors[class_id])
            draw_caption(color_image, (xmin, ymin), caption)

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
