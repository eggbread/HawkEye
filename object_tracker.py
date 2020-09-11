import time, random
import numpy as np
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs, convert_boxes

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

flags.DEFINE_string('classes', './data/labels/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './weights/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('video', './data/video/test.mp4',
                    'path to video file or number for webcam)')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')


class HawkEye():
    def __init__(self):
        self.y_axis = -1
        self.clickFlag = False

        self.max_cosine_distance = 0.5
        nn_budget = None
        self.nms_max_overlap = 1.0
        model_filename = 'model_data/mars-small128.pb'
        self.encoder = gdet.create_box_encoder(model_filename, batch_size=1)
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", self.max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.y_axis = y

    def main(self, _argv):
        if FLAGS.tiny:
            yolo = YoloV3Tiny(classes=FLAGS.num_classes)
        else:
            yolo = YoloV3(classes=FLAGS.num_classes)

        yolo.load_weights(FLAGS.weights)
        logging.info('weights loaded')

        class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
        logging.info('classes loaded')

        try:
            vid = cv2.VideoCapture(int(FLAGS.video))
        except:
            vid = cv2.VideoCapture(FLAGS.video)

        out = None

        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.y_axis = height + 1
        if FLAGS.output:
            # by default VideoCapture returns float instead of int
            fps = int(vid.get(cv2.CAP_PROP_FPS))
            codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
            out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))
            list_file = open('detection.txt', 'w')
            frame_index = -1

        fps = 0.0
        count = 0

        cv2.namedWindow('HawkEye')
        cv2.setMouseCallback('HawkEye', self.mouse_callback)

        while True:
            _, img = vid.read()

            if img is None:
                logging.warning("Empty Frame")
                time.sleep(0.1)
                count += 1
                if count < 3:
                    continue
                else:
                    break

            if self.y_axis < height:
                cv2.line(img, (0, self.y_axis), (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), self.y_axis), (255, 0, 0), 3)

            img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_in = tf.expand_dims(img_in, 0)
            img_in = transform_images(img_in, FLAGS.size)

            t1 = time.time()
            boxes, scores, classes, nums = yolo.predict(img_in)

            classes = classes[0]
            names = []
            for i in range(len(classes)):
                names.append(class_names[int(classes[i])])
            names = np.array(names)
            converted_boxes = convert_boxes(img, boxes[0])
            features = self.encoder(img, converted_boxes)
            detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                          zip(converted_boxes, scores[0], names, features)]

            # initialize color map
            cmap = plt.get_cmap('tab20b')
            colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

            # run non-maxima suppresion
            boxs = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            classes = np.array([d.class_name for d in detections])
            indices = preprocessing.non_max_suppression(boxs, classes, self.nms_max_overlap, scores)
            detections = [detections[i] for i in indices]

            # Call the tracker
            self.tracker.predict()
            self.tracker.update(detections)
            for track in self.tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                bbox = track.to_tlbr()
                class_name = track.get_class()
                color = colors[int(track.track_id) % len(colors)]
                color = [i * 255 for i in color]
                present_x, present_y, w, h = track.to_xywh()
                present_size = int(w * h)

                if self.y_axis <= present_y:
                    if track.size < present_size and track.y_axis < self.y_axis:
                        label = 'coming'
                    else:
                        label = 'warning'
                else:
                    label = ''
                cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.putText(img, label, (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_PLAIN, 2,
                            (255, 255, 255), 2)

            # print fps on screen
            fps = (fps + (1. / (time.time() - t1))) / 2
            cv2.putText(img, "FPS: {:.2f}".format(fps), (0, 30),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
            cv2.imshow('HawkEye', img)
            if FLAGS.output:
                out.write(img)
                frame_index = frame_index + 1
                list_file.write(str(frame_index) + ' ')
                if len(converted_boxes) != 0:
                    for i in range(0, len(converted_boxes)):
                        list_file.write(str(converted_boxes[i][0]) + ' ' + str(converted_boxes[i][1]) + ' ' + str(
                            converted_boxes[i][2]) + ' ' + str(converted_boxes[i][3]) + ' ')
                list_file.write('\n')

            # press q to quit
            if cv2.waitKey(1) == ord('q'):
                break
        vid.release()
        if FLAGS.output:
            out.release()
            list_file.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        hawkEye = HawkEye()
        app.run(hawkEye.main)
    except SystemExit:
        pass
