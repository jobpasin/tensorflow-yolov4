import colorsys
import random

import cv2
import numpy as np
from shutil import copyfile
import tensorflow as tf
import csv
import os
import math

try:
    from tensorflow_yolov4.core.config import cfg
except ModuleNotFoundError:
    from core.config import cfg


def load_freeze_layer(model='yolov4', tiny=False):
    if tiny:
        if model == 'yolov3':
            freeze_layouts = ['conv2d_9', 'conv2d_12']
        else:
            freeze_layouts = ['conv2d_17', 'conv2d_20']
    else:
        if model == 'yolov3':
            freeze_layouts = ['conv2d_58', 'conv2d_66', 'conv2d_74']
        else:
            freeze_layouts = ['conv2d_93', 'conv2d_101', 'conv2d_109']
    return freeze_layouts


def load_weights(model, weights_file, model_name='yolov4', is_tiny=False):
    if is_tiny:
        if model_name == 'yolov3':
            layer_size = 13
            output_pos = [9, 12]
        else:
            layer_size = 21
            output_pos = [17, 20]
    else:
        if model_name == 'yolov3':
            layer_size = 75
            output_pos = [58, 66, 74]
        else:
            layer_size = 110
            output_pos = [93, 101, 109]
    wf = open(weights_file, 'rb')
    major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)

    j = 0
    for i in range(layer_size):
        conv_layer_name = 'conv2d_%d' % i if i > 0 else 'conv2d'
        bn_layer_name = 'batch_normalization_%d' % j if j > 0 else 'batch_normalization'

        conv_layer = model.get_layer(conv_layer_name)
        filters = conv_layer.filters
        k_size = conv_layer.kernel_size[0]
        in_dim = conv_layer.input_shape[-1]

        if i not in output_pos:
            # darknet weights: [beta, gamma, mean, variance]
            bn_weights = np.fromfile(wf, dtype=np.float32, count=4 * filters)
            # tf weights: [gamma, beta, mean, variance]
            bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
            bn_layer = model.get_layer(bn_layer_name)
            j += 1
        else:
            conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)

        # darknet shape (out_dim, in_dim, height, width)
        conv_shape = (filters, in_dim, k_size, k_size)
        conv_weights = np.fromfile(wf, dtype=np.float32, count=np.product(conv_shape))
        # tf shape (height, width, in_dim, out_dim)
        conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])

        if i not in output_pos:
            conv_layer.set_weights([conv_weights])
            bn_layer.set_weights(bn_weights)
        else:
            conv_layer.set_weights([conv_weights, conv_bias])

    # assert len(wf.read()) == 0, 'failed to read all data'
    wf.close()


def read_class_names(class_file_name):
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names


def load_config(FLAGS):
    if FLAGS.tiny:
        STRIDES = np.array(cfg.YOLO.STRIDES_TINY)
        ANCHORS = get_anchors(cfg.YOLO.ANCHORS_TINY, FLAGS.tiny)
        XYSCALE = cfg.YOLO.XYSCALE_TINY if FLAGS.model == 'yolov4' else [1, 1]
    else:
        STRIDES = np.array(cfg.YOLO.STRIDES)
        if FLAGS.model == 'yolov4':
            ANCHORS = get_anchors(cfg.YOLO.ANCHORS, FLAGS.tiny)
        elif FLAGS.model == 'yolov3':
            ANCHORS = get_anchors(cfg.YOLO.ANCHORS_V3, FLAGS.tiny)
        XYSCALE = cfg.YOLO.XYSCALE if FLAGS.model == 'yolov4' else [1, 1, 1]
    NUM_CLASS = len(read_class_names(cfg.YOLO.CLASSES))

    return STRIDES, ANCHORS, NUM_CLASS, XYSCALE


def get_anchors(anchors_path, tiny=False):
    anchors = np.array(anchors_path)
    if tiny:
        return anchors.reshape(2, 3, 2)
    else:
        return anchors.reshape(3, 3, 2)


def image_preprocess(image, target_size, gt_boxes=None):
    ih, iw = target_size
    h, w, _ = image.shape

    scale = min(iw / w, ih / h)
    nw, nh = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih - nh) // 2
    image_paded[dh:nh + dh, dw:nw + dw, :] = image_resized
    image_paded = image_paded / 255.

    if gt_boxes is None:
        return image_paded

    else:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_paded, gt_boxes


def is_inside(boundary_matrix, point, width=500, height=500):
    point = np.array([[point[0]], [point[1]], [1]])
    new_point = np.matmul(boundary_matrix, point)
    new_point = new_point / new_point[2]
    if 0 <= new_point[0] <= width and 0 <= new_point[1] <= height:
        return True
    return False


def is_inside_bbox(boundary_matrix, startX, endX, startY, endY, width=500, height=500):
    return is_inside(boundary_matrix, np.array([startX, startY]), width, height) or \
           is_inside(boundary_matrix, np.array([startX, endY]), width, height) or \
           is_inside(boundary_matrix, np.array([endX, startY]), width, height) or \
           is_inside(boundary_matrix, np.array([endX, endY]), width, height)


# New: Use for rerun_from_log.py
def draw_bbox(image, bboxes, classes=None, show_label=True, boundary=None):
    if classes is None:
        classes = read_class_names(cfg.YOLO.CLASSES)
    num_classes = len(classes)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    out_boxes, out_scores, out_classes, num_boxes = bboxes
    for i in range(num_boxes[0]):
        if int(out_classes[0][i]) < 0 or int(out_classes[0][i]) > num_classes: continue
        coor = out_boxes[0][i]
        if boundary is not None:  # Add special condition to show bbox if centroid is inside roi
            centroid = np.array([(coor[1] + coor[3]) / 2, (coor[0] + coor[2]) / 2])
            in_roi = is_inside(boundary.warp_matrix, centroid, boundary.width, boundary.height)
            # in_roi = is_inside_bbox(boundary.warp_matrix, coor[1], coor[3], coor[0], coor[2], boundary.width,
            #                         boundary.height)
            if not in_roi: continue
        coor[0] = int(coor[0] * image_h)
        coor[2] = int(coor[2] * image_h)
        coor[1] = int(coor[1] * image_w)
        coor[3] = int(coor[3] * image_w)

        fontScale = 0.5
        score = out_scores[0][i]
        class_ind = int(out_classes[0][i])
        bbox_color = colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        c1, c2 = (int(coor[1]), int(coor[0])), (int(coor[3]), int(coor[2]))
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

        if show_label:
            bbox_mess = '%s: %.2f' % (classes[class_ind], score)
            t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
            c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
            cv2.rectangle(image, c1, (np.float32(c3[0]), np.float32(c3[1])), bbox_color, -1)  # filled

            cv2.putText(image, bbox_mess, (c1[0], np.float32(c1[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)
    return image


def bbox_iou(bboxes1, bboxes2):
    """
    @param bboxes1: (a, b, ..., 4)
    @param bboxes2: (A, B, ..., 4)
        x:X is 1:n or n:n or n:1
    @return (max(a,A), max(b,B), ...)
    ex) (4,):(3,4) -> (3,)
        (2,1,4):(2,3,4) -> (2,3)
    """
    bboxes1_area = bboxes1[..., 2] * bboxes1[..., 3]
    bboxes2_area = bboxes2[..., 2] * bboxes2[..., 3]

    bboxes1_coor = tf.concat(
        [
            bboxes1[..., :2] - bboxes1[..., 2:] * 0.5,
            bboxes1[..., :2] + bboxes1[..., 2:] * 0.5,
        ],
        axis=-1,
    )
    bboxes2_coor = tf.concat(
        [
            bboxes2[..., :2] - bboxes2[..., 2:] * 0.5,
            bboxes2[..., :2] + bboxes2[..., 2:] * 0.5,
        ],
        axis=-1,
    )

    left_up = tf.maximum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
    right_down = tf.minimum(bboxes1_coor[..., 2:], bboxes2_coor[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]

    union_area = bboxes1_area + bboxes2_area - inter_area

    iou = tf.math.divide_no_nan(inter_area, union_area)

    return iou


def bbox_giou(bboxes1, bboxes2):
    """
    Generalized IoU
    @param bboxes1: (a, b, ..., 4)
    @param bboxes2: (A, B, ..., 4)
        x:X is 1:n or n:n or n:1
    @return (max(a,A), max(b,B), ...)
    ex) (4,):(3,4) -> (3,)
        (2,1,4):(2,3,4) -> (2,3)
    """
    bboxes1_area = bboxes1[..., 2] * bboxes1[..., 3]
    bboxes2_area = bboxes2[..., 2] * bboxes2[..., 3]

    bboxes1_coor = tf.concat(
        [
            bboxes1[..., :2] - bboxes1[..., 2:] * 0.5,
            bboxes1[..., :2] + bboxes1[..., 2:] * 0.5,
        ],
        axis=-1,
    )
    bboxes2_coor = tf.concat(
        [
            bboxes2[..., :2] - bboxes2[..., 2:] * 0.5,
            bboxes2[..., :2] + bboxes2[..., 2:] * 0.5,
        ],
        axis=-1,
    )

    left_up = tf.maximum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
    right_down = tf.minimum(bboxes1_coor[..., 2:], bboxes2_coor[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]

    union_area = bboxes1_area + bboxes2_area - inter_area

    iou = tf.math.divide_no_nan(inter_area, union_area)

    enclose_left_up = tf.minimum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
    enclose_right_down = tf.maximum(
        bboxes1_coor[..., 2:], bboxes2_coor[..., 2:]
    )

    enclose_section = enclose_right_down - enclose_left_up
    enclose_area = enclose_section[..., 0] * enclose_section[..., 1]

    giou = iou - tf.math.divide_no_nan(enclose_area - union_area, enclose_area)

    return giou


def bbox_ciou(bboxes1, bboxes2):
    """
    Complete IoU
    @param bboxes1: (a, b, ..., 4)
    @param bboxes2: (A, B, ..., 4)
        x:X is 1:n or n:n or n:1
    @return (max(a,A), max(b,B), ...)
    ex) (4,):(3,4) -> (3,)
        (2,1,4):(2,3,4) -> (2,3)
    """
    bboxes1_area = bboxes1[..., 2] * bboxes1[..., 3]
    bboxes2_area = bboxes2[..., 2] * bboxes2[..., 3]

    bboxes1_coor = tf.concat(
        [
            bboxes1[..., :2] - bboxes1[..., 2:] * 0.5,
            bboxes1[..., :2] + bboxes1[..., 2:] * 0.5,
        ],
        axis=-1,
    )
    bboxes2_coor = tf.concat(
        [
            bboxes2[..., :2] - bboxes2[..., 2:] * 0.5,
            bboxes2[..., :2] + bboxes2[..., 2:] * 0.5,
        ],
        axis=-1,
    )

    left_up = tf.maximum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
    right_down = tf.minimum(bboxes1_coor[..., 2:], bboxes2_coor[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]

    union_area = bboxes1_area + bboxes2_area - inter_area

    iou = tf.math.divide_no_nan(inter_area, union_area)

    enclose_left_up = tf.minimum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
    enclose_right_down = tf.maximum(
        bboxes1_coor[..., 2:], bboxes2_coor[..., 2:]
    )

    enclose_section = enclose_right_down - enclose_left_up

    c_2 = enclose_section[..., 0] ** 2 + enclose_section[..., 1] ** 2

    center_diagonal = bboxes2[..., :2] - bboxes1[..., :2]

    rho_2 = center_diagonal[..., 0] ** 2 + center_diagonal[..., 1] ** 2

    diou = iou - tf.math.divide_no_nan(rho_2, c_2)

    v = (
                (
                        tf.math.atan(
                            tf.math.divide_no_nan(bboxes1[..., 2], bboxes1[..., 3])
                        )
                        - tf.math.atan(
                    tf.math.divide_no_nan(bboxes2[..., 2], bboxes2[..., 3])
                )
                )
                * 2
                / np.pi
        ) ** 2

    alpha = tf.math.divide_no_nan(v, 1 - iou + v)

    ciou = diou - alpha * v

    return ciou


def nms(bboxes, iou_threshold, sigma=0.3, method='nms'):
    """
    :param bboxes: (xmin, ymin, xmax, ymax, score, class)

    Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
          https://github.com/bharatsingh430/soft-nms
    """
    classes_in_img = list(set(bboxes[:, 5]))
    best_bboxes = []

    for cls in classes_in_img:
        cls_mask = (bboxes[:, 5] == cls)
        cls_bboxes = bboxes[cls_mask]

        while len(cls_bboxes) > 0:
            max_ind = np.argmax(cls_bboxes[:, 4])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
            iou = bbox_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            weight = np.ones((len(iou),), dtype=np.float32)

            assert method in ['nms', 'soft-nms']

            if method == 'nms':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0

            if method == 'soft-nms':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))

            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > 0.
            cls_bboxes = cls_bboxes[score_mask]

    return best_bboxes


def freeze_all(model, frozen=True):
    model.trainable = not frozen
    if isinstance(model, tf.keras.Model):
        for l in model.layers:
            freeze_all(l, frozen)


def unfreeze_all(model, frozen=False):
    model.trainable = not frozen
    if isinstance(model, tf.keras.Model):
        for l in model.layers:
            unfreeze_all(l, frozen)


def convert_csv_to_darknet_label(csv_path, image_path, result_path, result_text_path, class_name_list, only_label=True):
    """
     "Convert .csv file (Using dataset.py) into darknet label format (.txt)"
     @param csv_path: Path to csv file containing file name, coordinates (expect OpenImage format)
     @param image_path: Path to directory of input images
     @param result_path: Path to directory of output image + text file(only those exist in csv path will be copied here)
     @param result_text_path: Path to produce .txt file containing name of all images used for darknet format
     @param class_name_list: List of name of class that will be kept. Otherwise, that bounding box will be ignored
     @param only_label: If True, it will skip any image that does not contain class_name_list. Good for filter positive images

    """

    if not os.path.exists(result_path):
        os.mkdir(result_path)
    if len(os.listdir(result_path)) > 0:
        print("Warning, Result path is not empty")
    class_name_list = [x.lower() for x in class_name_list]
    count = 0
    folder_name = result_path.split('/')[-1]
    with open(result_text_path, mode='w') as write_file:
        with open(csv_path, mode='r') as csv_file:
            csv_reader = csv.reader(csv_file)
            header = next(csv_reader)
            for row in csv_reader:
                image_name = row[0]
                class_name = row[1].strip().lower()
                try:
                    class_id = class_name_list.index(class_name)
                except ValueError:
                    if only_label:
                        continue
                    else:
                        with open(os.path.join(result_path, image_name + ".txt"), mode='a') as txt_writer:
                            pass
                        original_image_dir = os.path.join(image_path, image_name + ".jpg")
                        new_image_dir = os.path.join(result_path, image_name + ".jpg")
                        if not os.path.exists(new_image_dir):
                            copyfile(original_image_dir, new_image_dir)

                xmin = float(row[5])
                xmax = float(row[6])
                ymin = float(row[7])
                ymax = float(row[8])
                xcenter = (xmin + xmax) / 2
                ycenter = (ymin + ymax) / 2
                width = xmax - xmin
                height = ymax - ymin
                if (xcenter <= 0 or ycenter <= 0 or width <= 0 or height <= 0 or
                        xcenter > 1 or ycenter > 1 or width > 1 or height > 1):
                    print("Warning, image {} have invalid size".format(image_name))

                with open(os.path.join(result_path, image_name + ".txt"), mode='a') as txt_writer:
                    txt_writer.write("{} {} {} {} {}\n".format(class_id, xcenter, ycenter, width, height))
                original_image_dir = os.path.join(image_path, image_name + ".jpg")
                new_image_dir = os.path.join(result_path, image_name + ".jpg")
                if not os.path.exists(new_image_dir):
                    copyfile(original_image_dir, new_image_dir)
                    write_file.write(os.path.join("data", folder_name, image_name + ".jpg\n"))
                if count % 500 == 0:
                    print("Copy: {} images".format(count), end='\r')
                count += 1


def filter_class_copy_image(image_path, new_image_path, in_csv_file, out_csv_file, class_name_list, filter=None):
    if not os.path.exists(new_image_path):
        os.mkdir(new_image_path)
    if len(os.listdir(new_image_path)) > 0:
        raise ValueError("Result path is not empty")
    class_name_list = [x.lower() for x in class_name_list]
    if filter is None:
        filter = {}
    with open(in_csv_file, mode='r') as csv_file:
        with open(out_csv_file, mode='w', newline='') as writer:
            csv_reader = csv.reader(csv_file)
            header = next(csv_reader)
            csv_writer = csv.DictWriter(writer, fieldnames=header)
            csv_writer.writeheader()
            for row in csv_reader:
                image_name = row[0]
                class_name = row[1].lower()
                if class_name in class_name_list:
                    class_name = class_name
                elif class_name in filter.keys():
                    class_name = filter[class_name]
                else:
                    continue
                id = row[2]
                label_id = row[3]
                confidence = row[4]
                xmin = row[5]
                xmax = row[6]
                ymin = row[7]
                ymax = row[8]
                data = {'ImageID': image_name, 'LabelName': class_name, 'Label_ID': class_name,
                        'Confidence': confidence, 'XMin': xmin, 'XMax': xmax, 'YMin': ymin,
                        'YMax': ymax, 'id': id}
                original_image_dir = os.path.join(image_path, image_name + ".jpg")
                new_image_dir = os.path.join(new_image_path, image_name + ".jpg")
                if not os.path.exists(original_image_dir):
                    print("Warning, image: {} not found".format(original_image_dir))
                    continue
                if not os.path.exists(new_image_dir):
                    copyfile(original_image_dir, new_image_dir)
                csv_writer.writerow(data)


def combine_text():
    """
    Combine multiple text file (as a element of input_path) plus one negative text file (negative_path)
    into a single output (output_path)
    """
    base_path = "F:project/darknet_build/data"

    input_path = ["train_name.txt", "4-001.txt", "4-002.txt", "4-003.txt", "4-004.txt", "4-005.txt",
                  "4-006.txt", "4-007.txt", "4-008.txt", "4-009.txt", "4-010.txt", "4-011.txt"]
    negative_path = "train_negative_name.txt"
    output_path = "train_name_combined2.txt"

    with open(os.path.join(base_path, output_path), 'w') as write_file:
        count = 0
        for i in input_path:
            with open(os.path.join(base_path, i), 'r') as read_file:
                for data in read_file:
                    data = data.replace('\\', '/')
                    write_file.write(data)
                    count += 1

        count = (math.ceil(count / 1000) + 1) * 1000
        with open(os.path.join(base_path, negative_path), 'r') as read_file:
            negative_count = 0
            for data in read_file:
                data = data.replace('\\', '/')
                write_file.write(data)
                if negative_count == count:
                    break
                negative_count += 1
            if negative_count < count:
                print("Warning, negative image not enough")
        print("Getting {} negative images".format(count))

    with open(os.path.join(base_path, "validation_name_combined2.txt"), 'w') as write_file:
        count = 0
        with open(os.path.join(base_path, "validation_name.txt"), 'r') as read_file:
            for data in read_file:
                data = data.replace('\\', '/')
                write_file.write(data)
                count += 1

        with open(os.path.join(base_path, "validation_negative_name.txt"), 'r') as read_file:
            negative_count = 0
            for data in read_file:
                data = data.replace('\\', '/')
                write_file.write(data)
                # if negative_count == count:
                #     break
                negative_count += 1

        print("Getting {} validation negative images".format(count))


if __name__ == "__main__":
    dataset = "train"
    csv_path = "F:/project/openimage_dataset/new_label/vehicle-{}-annotation-bbox_sum.csv".format(dataset)
    image_path = "F:/project/openimage_dataset/vehicle/{}_vehicle_image/".format(dataset)
    result_path = "F:/project/darknet_build/data/vehicle-{}".format(dataset)
    result_path_text = "F:/project/darknet_build/data/{}_name.txt".format(dataset)
    # class_name_list = ["car", "motorcycle", "pickup", "passenger car", "bus", "truck", "trailer truck"]
    class_name_list = ["pc", "m", "p", "7up", "b", "t", "tt"]
    # class_name_list = ['Bus', 'Motorcycle', 'Truck', 'Vehicle', 'Van', 'Car', 'Taxi']
    # class_name_list = ['Bus', 'Land vehicle', 'Motorcycle', 'Truck', 'Vehicle', 'Van', 'Car', 'Taxi']
    convert_csv_to_darknet_label(csv_path, image_path, result_path, result_path_text, class_name_list, only_label=True)

    image_path = "F:/project/openimage_dataset/vehicle/train_vehicle_image/train_00"
    new_image_path = "F:/project/openimage_dataset/train_vehicle_image_debug"
    in_csv_file = "F:/project/openimage_dataset/train_debug.csv"
    out_csv_file = "F:/project/openimage_dataset/train_debug_output.csv"
    filter = {'c': "car", 'm': "motorcycle", "p": "pickup", "pc": "passenger car", "b": "bus", "t": "truck",
              "tt": "trailer"}
    # filter_class_copy_image(image_path, new_image_path, in_csv_file, out_csv_file, class_name_list, filter=filter)
