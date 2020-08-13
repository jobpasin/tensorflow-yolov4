import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
import tensorflow_yolov4.core.utils as utils
from tensorflow_yolov4.core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import time


class YoloV4:
    def __init__(self, FLAGS, interested_class=None):
        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)
        self.FLAGS = FLAGS
        self.interested_class = interested_class
        self.interpreter = None
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
            batch_data = tf.constant(image_data)
            start_time = time.time()
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

    def draw_bbox(self, frame, pred_bbox):
        return utils.draw_bbox(frame, pred_bbox, show_label=True)

    def filter_class(self, pred_bbox):
        new_boxes = []
        new_scores = []
        new_classes = []
        current_detections = 0
        for i in range(pred_bbox[3][0]):
            if int(pred_bbox[2][0, i]) in self.interested_class:
                new_boxes.append(pred_bbox[0][0, i, :])
                new_scores.append(pred_bbox[1][0, i])
                new_classes.append(pred_bbox[2][0, i])
                current_detections += 1
        if current_detections == 0:
            return [np.zeros([1, 0, 4]), np.zeros([1, 0]), np.zeros([1, 0]), np.array([current_detections])]
        return [np.expand_dims(np.stack(new_boxes), axis=0), np.expand_dims(np.stack(new_scores), axis=0),
                np.expand_dims(np.stack(new_classes), axis=0), np.array([current_detections])]
