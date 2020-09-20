from ctypes import *
import random
import os
import cv2
import time
import darknet
import argparse
from threading import Thread, enumerate
from queue import Queue

from sort import *
import numpy as np


def parser():
    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    parser.add_argument("--input", type=str, default="/home/eggbread/Videos/oxford.mp4",
                        help="video source. If empty, uses webcam 0 stream")
    parser.add_argument("--out_filename", type=str, default="",
                        help="inference video name. Not saved if empty")
    parser.add_argument("--weights", default="yolov4.weights",
                        help="yolo weights path")
    parser.add_argument("--dont_show", action='store_true',
                        help="windown inference display. For headless systems")
    parser.add_argument("--ext_output", action='store_true',
                        help="display bbox coordinates of detected objects")
    parser.add_argument("--config_file", default="./cfg/yolov4.cfg",
                        help="path to config file")
    parser.add_argument("--data_file", default="./cfg/coco.data",
                        help="path to data file")
    parser.add_argument("--thresh", type=float, default=.25,
                        help="remove detections with confidence below this value")
    return parser.parse_args()

def str2int(video_path):
    """
    argparse returns and string althout webcam uses int (0, 1 ...)
    Cast to int if needed
    """
    try:
        return int(video_path)
    except ValueError:
        return video_path

def check_arguments_errors(args):
    assert 0 < args.thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(args.config_file):
        raise(ValueError("Invalid config path {}".format(os.path.abspath(args.config_file))))
    if not os.path.exists(args.weights):
        raise(ValueError("Invalid weight path {}".format(os.path.abspath(args.weights))))
    if not os.path.exists(args.data_file):
        raise(ValueError("Invalid data file path {}".format(os.path.abspath(args.data_file))))
    if str2int(args.input) == str and not os.path.exists(args.input):
        raise(ValueError("Invalid video path {}".format(os.path.abspath(args.input))))

class HawkEye(object):
    def __init__(self, height):
        self.y_axis = height

    def set_saved_video(self, input_video, output_video, size):
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        fps = int(input_video.get(cv2.CAP_PROP_FPS))
        video = cv2.VideoWriter(output_video, fourcc, fps, size)
        return video


    def video_capture(self, frame_queue, darknet_image_queue):
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (width, height),
                                       interpolation=cv2.INTER_LINEAR)
            frame_queue.put(frame_resized)
            darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())
            darknet_image_queue.put(darknet_image)
        cap.release()

    def convert_xyxy(self, bbox):
        x, y, w, h = bbox
        xmin = round(x - (w / 2))
        xmax = round(x + (w / 2))
        ymin = round(y - (h / 2))
        ymax = round(y + (h / 2))
        return xmin, ymin, xmax, ymax

    def inference(self, darknet_image_queue, detections_queue, fps_queue):
        tracker = Sort()
        while cap.isOpened():
            darknet_image = darknet_image_queue.get()
            prev_time = time.time()
            detections = darknet.detect_image(network, class_names, darknet_image, thresh=args.thresh)

            detections = np.array(detections)
            dets = []
            for score, bbox in detections[:, 1:]:
                x1,y1,x2,y2 = self.convert_xyxy(bbox)
                dets.append([x1, y1, x2, y2, float(score) / 100])
            tracked_object = tracker.update(np.array(dets))
            detections_queue.put(tracked_object)

            fps = int(1/(time.time() - prev_time))
            fps_queue.put(fps)
            print("FPS: {}".format(fps))
            darknet.print_detections(detections, args.ext_output)
        cap.release()

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.y_axis = y

    def drawing(self, frame_queue, detections_queue, fps_queue):
        random.seed(3)  # deterministic bbox colors
        video = self.set_saved_video(cap, args.out_filename, (width, height))

        cv2.namedWindow('HawkEye')
        cv2.setMouseCallback('HawkEye', self.mouse_callback)

        while cap.isOpened():
            frame_resized = frame_queue.get()
            detections = detections_queue.get()
            fps = fps_queue.get()
            if self.y_axis < height:
                cv2.line(frame_resized, (0, self.y_axis), (width, self.y_axis), (255, 0, 0), 3)
            if frame_resized is not None:
                image = darknet.draw_boxes(detections, frame_resized, class_colors, self.y_axis)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                if args.out_filename is not None:
                    video.write(image)
                if not args.dont_show:
                    cv2.imshow('HawkEye', image)
                if cv2.waitKey(fps) == 27:
                    break
        cap.release()
        video.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    frame_queue = Queue()
    darknet_image_queue = Queue(maxsize=1)
    detections_queue = Queue(maxsize=1)
    fps_queue = Queue(maxsize=1)

    args = parser()
    check_arguments_errors(args)
    network, class_names, class_colors = darknet.load_network(
            args.config_file,
            args.data_file,
            args.weights,
            batch_size=1
        )
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)
    input_path = str2int(args.input)
    cap = cv2.VideoCapture(input_path)
    run = HawkEye(height)
    Thread(target=run.video_capture, args=(frame_queue, darknet_image_queue)).start()
    Thread(target=run.inference, args=(darknet_image_queue, detections_queue, fps_queue)).start()
    Thread(target=run.drawing, args=(frame_queue, detections_queue, fps_queue)).start()
