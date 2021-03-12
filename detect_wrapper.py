import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import time
import sys
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.python.saved_model import tag_constants
import tensorflow_yolov4.core.utils as utils
import tensorflow_yolov4.core.config as config
from absl import logging

from queue import Queue, Empty, Full
from performance import TimeMeasure

# from tensorflow_yolov4.core.yolov4 import filter_boxes
sys.stdout.flush()


class YoloV4:
    def __init__(self, FLAGS, interested_class=None):
        logging.set_verbosity(logging.WARNING)
        # config = ConfigProto()
        # config.gpu_options.allow_growth = True
        # session = InteractiveSession(config=config)
        # tf.debugging.set_log_device_placement(True)
        gpus = tf.config.experimental.list_physical_devices('GPU')
        try:
            # Currently, memory growth needs to be the same across GPUs
            assert FLAGS.gpu < len(gpus), "--gpu is higher than number of gpu available. " \
                                          "(Choose integer less than or equal to {} or use -1)".format(len(gpus) - 1)
            if FLAGS.gpu != -1:
                tf.config.experimental.set_visible_devices(gpus[FLAGS.gpu], 'GPU')
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            logging.debug("{} Physical GPUS, {} Logical GPUS".format(len(gpus), len(logical_gpus)))
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

        if len(gpus) == 0:
            logging.warning('No GPU found')
        self.strategy = tf.distribute.MirroredStrategy()

        self.FLAGS = FLAGS
        self.interested_class = interested_class
        self.interpreter = None

        with self.strategy.scope():
            self.saved_model_loaded = None
            self.infer = None
            if FLAGS.framework == 'tflite':
                self.interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                print("input details: ", self.input_details)
                print("output details: ", self.output_details)
            else:
                self.saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
                self.infer = self.saved_model_loaded.signatures['serving_default']
        if FLAGS.debug:
            logging.set_verbosity(logging.DEBUG)
        else:
            logging.set_verbosity(logging.INFO)

    def predict_wrapper(self, in_queue: Queue, out_queue: Queue, batch_size, stop_event):
        if self.FLAGS.framework == 'tflite':
            raise ValueError("")
        image_buffer, frame_id_buffer, image_copy_buffer = [], [], []
        STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(self.FLAGS)
        input_size = self.FLAGS.size
        stop_flag = False
        while not stop_flag:
            if stop_event.isSet():  # Clear data in queue if receiving stop signal
                with in_queue.mutex:
                    in_queue.queue.clear()
                break
            # if in_queue.qsize() == 0:
            #     continue
            # Fetching data
            try:
                frame, frame_id = in_queue.get(timeout=15)
            except Empty:
                logging.error("Yolo thread timeout")
                break

            # Add data into the list until the list has [batch_size] images
            if frame is None:
                stop_flag = True
            else:
                frame_size = frame.shape[:2]
                image_copy_buffer.append(frame.copy())
                image_data = cv2.resize(frame, (input_size, input_size))
                image_data = image_data / 255.
                # image_data = image_data[np.newaxis, ...].astype(np.float32)
                image_buffer.append(image_data)
                frame_id_buffer.append(frame_id)

            # Run detection when we have [batch_size] images, or has stop_flag (finish program)
            if len(image_buffer) == batch_size or stop_flag:
                if len(image_buffer) > 0:
                    image_data = np.stack(image_buffer).astype(np.float32)
                    batch_data = tf.constant(image_data)
                    pred_bbox = self.infer(batch_data)
                    for key, value in pred_bbox.items():
                        boxes = value[:, :, 0:4]
                        pred_conf = value[:, :, 4:]
                    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                        scores=tf.reshape(
                            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                        max_output_size_per_class=50,
                        max_total_size=50,
                        iou_threshold=self.FLAGS.iou,
                        score_threshold=self.FLAGS.score
                    )
                    pred_bboxes = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
                    if self.interested_class is not None:  # Filter only prediction of interested class
                        pred_bboxes = self.filter_class(pred_bboxes)

                    # Send prediction one by one
                    for i in range(np.shape(pred_bboxes[3])[0]):
                        pred_bbox = [pred_bboxes[0][i:i + 1, :, :], pred_bboxes[1][i:i + 1, :],
                                     pred_bboxes[2][i:i + 1, :], pred_bboxes[3][i:i + 1]]
                        if not stop_event.isSet():
                            try:
                                out_queue.put([pred_bbox, frame_id_buffer[i], image_copy_buffer[i]], timeout=25)
                            except Full:
                                logging.error("bbox_queue: Read vid timeout")
                                break
                # Reset buffer after predictions
                image_buffer, frame_id_buffer, image_copy_buffer = [], [], []
        if stop_flag:
            try:
                out_queue.put([None, None, None], timeout=25)
            except Full:
                pass
        logging.debug("Yolo thread complete")

    def predict(self, frame):
        STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(self.FLAGS)
        input_size = self.FLAGS.size
        video_path = self.FLAGS.video
        # frame_id = 0

        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        if self.FLAGS.framework == 'tflite':
            self.interpreter.set_tensor(self.input_details[0]['index'], image_data)
            self.interpreter.invoke()
            pred = [self.interpreter.get_tensor(self.output_details[i]['index']) for i in
                    range(len(self.output_details))]
            if self.FLAGS.model == 'yolov3' and self.FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data, name='batch_data')
            pred_bbox = self.infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]
        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=self.FLAGS.iou,
            score_threshold=self.FLAGS.score
        )
        pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
        if self.interested_class is not None:
            pred_bbox = self.filter_class(pred_bbox)
        return pred_bbox

    def filter_class(self, pred_bbox):
        boxes_new, scores_new, classes_new, valid_detection_new = [], [], [], []
        num_detection = np.shape(pred_bbox[2])[1]
        num_batch = np.shape(pred_bbox[3])[0]
        for batch in range(num_batch):
            boxes_temp, scores_temp, classes_temp = [], [], []
            current_detections = 0
            for i in range(pred_bbox[3][batch]):
                if int(pred_bbox[2][batch, i]) in self.interested_class:
                    boxes_temp.append(pred_bbox[0][batch, i, :])
                    scores_temp.append(pred_bbox[1][batch, i])
                    classes_temp.append(pred_bbox[2][batch, i])
                    current_detections += 1
            for _ in range(num_detection - current_detections):
                boxes_temp.append(np.zeros([4]))
                scores_temp.append(0.0)
                classes_temp.append(0.0)
            # if current_detections == 0:
            #     boxes_new.append(np.zeros([1,4]))
            #     scores_new.append(np.zeros([1]))
            #     classes_new.append(np.zeros([0]))
            #     valid_detection_new.append(current_detections)
            #     # return ([np.zeros([1, 0, 4]), np.zeros([1, 0]), np.zeros([1, 0]), np.array([current_detections])])
            # else:
            boxes_new.append(np.stack(boxes_temp))
            scores_new.append(np.stack(scores_temp))
            classes_new.append(np.stack(classes_temp))
            valid_detection_new.append(current_detections)
            # return [np.expand_dims(np.stack(boxes_temp), axis=0), np.expand_dims(np.stack(scores_temp), axis=0),
            #         np.expand_dims(np.stack(classes_temp), axis=0), np.array([current_detections])]
        return [np.stack(boxes_new), np.stack(scores_new), np.stack(classes_new), np.array(valid_detection_new)]


def draw_bbox(frame, pred_bbox, boundary=None, classes=None):
    """
    Draw bounding box on yolov4 output
    @param frame:
    @param pred_bbox: Direct output from yolov4
    @param boundary: WarpMatrix class. Show only bbox in boundary. Give None to show all bbox in the image
    @param classes: List of full class name
    @return:
    """
    return utils.draw_bbox(frame, pred_bbox, classes=classes, show_label=True, boundary=boundary)


def get_class_name(dataset_type):
    if dataset_type == "coco":
        path = os.path.join("tensorflow_yolov4", "data/classes/coco.names")
    elif dataset_type == "doh":
        path = os.path.join("tensorflow_yolov4", "data/classes/vehicle.names")
    else:
        raise ValueError("dataset argument is invalid")

    assert os.path.exists(path), "Error: Does not find classes name file: {}".format(path)
    name = []
    with open(path, 'r') as f:
        for line in f:
            name.append(line.rstrip('\n'))
    if dataset_type == "coco":
        interested_class = [1, 2, 3, 5, 7]  # /Person/, Bicycle, Car, motorbike, bus, /train/, truck
    elif dataset_type == "doh":
        interested_class = [i for i in range(len(name))]
    return name, interested_class
