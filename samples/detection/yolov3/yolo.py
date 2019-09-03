import time
import sys
import os
import google.protobuf as pb
#import google.protobuf.text_format
from google.protobuf import text_format
import caffe
import cv2
from caffe.proto import caffe_pb2
from utils import letterbox_image

import numpy as np

np.set_printoptions(suppress=True)


class YOLO(object):
    def __init__(self, obj_threshold, nms_threshold, w, h, num_of_class, proto_path, model_path,
                 anchors=[[116,90, 156,198, 373,326], [30,61, 62,45, 59,119], [10,13, 16,30, 33,23]],
                 cali_proto="", cali_model="", int8_flag=0, int8_str="", gpu_device_id=0):
        """
        Args:
            obj_threshold:
            nms_threshold:
            w:
            h:
            num_of_class: number of class. Excluding "other"
            proto_path:
            model_path:
            anchors:
            cali_ptoto:
            cali_model:
            int8_flag:
            int8_str:
            gpu_device_id:
        """
        param = caffe_pb2.NetParameter()
        text_format.Merge(open(proto_path).read(), param)
        int8_layer = ""
        count = 0
        for layer in param.layer:
            int8_layer += str(layer.name)
            count += 1
            if str(layer.name) == int8_str:
                break
            if count != len(param.layer):
                int8_layer += ","

        self.obj_threshold = obj_threshold
        self.nms = nms_threshold
        self.nw = w
        self.nh = h
        self.num_of_class = num_of_class
        self.anchors = anchors
        caffe.set_mode_gpu()
        caffe.set_device(gpu_device_id)
        caffe.yolo()
        
        self.yolo = caffe.Net(proto_path, model_path, caffe.TEST)
        if int8_flag:
            self.yolo.int8_init(proto_path, cali_model, cali_proto, int8_layer)

    def process_feats(self, out, anchor):
        grid_size = out.shape[2]
        num_boxes_per_cell = 3
        
        out = np.transpose(out, (1, 2, 0))
        out = np.reshape(out, (grid_size, grid_size, num_boxes_per_cell, 5 + self.num_of_class))
        threshold_predictions = []
        
        anchors_tensor = np.array(anchor).reshape(1, 1, 3, 2)

        box_xy = self.sigmoid(out[..., :2])
        box_wh = np.exp(out[..., 2:4]) * anchors_tensor
        
        box_confidence = self.sigmoid(out[..., 4])
        box_confidence = np.expand_dims(box_confidence, axis=-1)
        box_class_probs = self.softmax(out[..., 5:])

        col = np.tile(np.arange(0, grid_size), grid_size).reshape(-1, grid_size)
        row = np.tile(np.arange(0, grid_size).reshape(-1, 1), grid_size)

        col = col.reshape(grid_size, grid_size, 1, 1).repeat(3, axis=-2)
        row = row.reshape(grid_size, grid_size, 1, 1).repeat(3, axis=-2)
        grid = np.concatenate((col, row), axis=-1)

        box_xy += grid
        box_xy /= (grid_size, grid_size)
        box_wh /= (self.nw, self.nh)

        boxes = np.concatenate((box_xy, box_wh), axis=-1)
        
        box_score = box_confidence * box_class_probs
        box_classes = np.argmax(box_score, axis=-1)
        box_class_score = np.max(box_score, axis=-1)

        pos = np.where(box_class_score >= self.obj_threshold)
 
        boxes = boxes[pos]
        scores = box_class_score[pos]
        scores = np.expand_dims(scores, axis=-1)
        classes = box_classes[pos]
        classes = np.expand_dims(classes, axis=-1)
        if boxes is not None:
            threshold_predictions = np.concatenate((boxes, scores, classes), axis=-1)

        return threshold_predictions

    def correct_boxes(self, predictions, image_shape):
        image_shape = np.array((image_shape[1], image_shape[0]))
        input_shape = np.array([float(self.nw), float(self.nh)])
        new_shape = np.floor(image_shape * np.min(input_shape / image_shape))
        offset = (input_shape - new_shape) / 2. / input_shape
        scale = input_shape / new_shape
        correct = []
        for prediction in predictions:
            x, y, w, h = prediction[0:4]
            box_xy = np.array([x, y])
            box_wh = np.array([w, h])
            score = prediction[4]
            cls = int(prediction[5])
            
            box_xy = (box_xy - offset) * scale
            box_wh = box_wh * scale

            box_xy = box_xy - box_wh / 2.
            box = np.concatenate((box_xy, box_wh), axis=-1)            
            box *= np.concatenate((image_shape, image_shape), axis=-1)
            correct.append([box, score, cls])
        return correct

    def softmax(self, x, t=-100.):
        x = x - np.max(x)
        if np.min(x) < t:
            x = x / np.min(x) * t
        exp_x = np.exp(x)
        out = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        return out

    def sigmoid(self, x):
        return 1. / (1. + np.exp(-x))

    def iou(self, box1, box2):
        inter_left_x = max(box1[0], box2[0])
        inter_left_y = max(box1[1], box2[1])
        inter_right_x = min(box1[0] + box1[2], box2[0] + box2[2])
        inter_right_y = min(box1[1] + box1[3], box2[1] + box2[3])
        
        if box1[0] == box2[0] and box1[1] == box2[1] and box1[2] == box2[2] and box1[3] == box2[3]:
            return 1.

        inter_w = max(0, inter_right_x - inter_left_x)
        inter_h = max(0, inter_right_y - inter_left_y)

        inter_area = inter_w * inter_h

        box1_area = box1[2] * box1[3]
        box2_area = box2[2] * box2[3]

        iou = inter_area / (box1_area + box2_area - inter_area)

        return iou

    def non_maximum_suppression(self, predictions, nms_threshold):
        nms_predictions = []
        nms_predictions.append(predictions[0])

        i = 1
        while i < len(predictions):
            nms_len = len(nms_predictions)
            keep = True
            j = 0
            while j < nms_len:
                current_iou = self.iou(predictions[i][0], nms_predictions[j][0])
                if current_iou > nms_threshold and predictions[i][2] == nms_predictions[j][2] and current_iou < 1.:
                    keep = False

                j = j + 1
            if keep:
                nms_predictions.append(predictions[i])
            i = i + 1

        return nms_predictions

    def yolo_out(self, out_feats, shape):
        outs = out_feats
        total_predictions = []
        for i, out in enumerate(outs):
            threshold_predictions = self.process_feats(out, self.anchors[i])
            total_predictions.extend(threshold_predictions)
        
        if not total_predictions:
            return None

        correct_predictions = self.correct_boxes(total_predictions, shape)
        
        correct_predictions.sort(key=lambda tup: tup[1], reverse=True)

        nms_predictions = self.non_maximum_suppression(correct_predictions, self.nms)
        return nms_predictions

    def predict(self, image_path):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        new_img = letterbox_image(img, self.nw, self.nh)

        self.yolo.blobs['data'].reshape(1, 3, self.nw, self.nh)
        self.yolo.blobs['data'].data[...] = new_img
        out_feats = self.yolo.forward()
        layer82_conv = out_feats['layer82-conv']
        layer94_conv = out_feats['layer94-conv']
        layer106_conv = out_feats['layer106-conv']

        batch_out = {}
        feat = [layer82_conv[0], layer94_conv[0], layer106_conv[0]]
        output = self.yolo_out(feat, img.shape)
        if not output:
            batch_out[0] = []
        else:
            batch_out[0] = output

        return batch_out
