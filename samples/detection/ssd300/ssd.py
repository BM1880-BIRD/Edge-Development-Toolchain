import time
import sys
import os
import google.protobuf as pb
#import google.protobuf.text_format
from google.protobuf import text_format
import caffe
from caffe.proto import caffe_pb2

import numpy as np

class SSD(object):
    def __init__(self, conf_threshold, w, h, proto_path, model_path,
                 cali_proto="", cali_model="", int8_flag=0, int8_str=""):
        """
        Args:
            obj_threshold:
            w:
            h:
            proto_path:
            model_path:
            cali_ptoto:
            cali_model:
            int8_flag:
            int8_str:
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

        self.conf_threshold = conf_threshold
        self.nw = w
        self.nh = h
        
        self.ssd = caffe.Net(proto_path, model_path, caffe.TEST)
        if int8_flag:
            self.ssd.int8_init(proto_path, cali_model, cali_proto, '')

    def parse_top_detection(self, resolution, detections, conf_threshold=0.6):
        det_label = detections[0, 0, :, 1]
        det_conf = detections[0, 0, :, 2]
        det_xmin = detections[0, 0, :, 3]
        det_ymin = detections[0, 0, :, 4]
        det_xmax = detections[0, 0, :, 5]
        det_ymax = detections[0, 0, :, 6]

        # Get detections with confidence higher than 0.6.
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= conf_threshold]

        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].astype(int).tolist()
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]

        bboxs = np.zeros((top_conf.shape[0], 4), dtype=int)
        for i in range(top_conf.shape[0]):
            bboxs[i][0] = int(round(top_xmin[i] * resolution[1]))
            bboxs[i][1] = int(round(top_ymin[i] * resolution[0]))
            bboxs[i][2] = int(round(top_xmax[i] * resolution[1])) - bboxs[i][0]
            bboxs[i][3] = int(round(top_ymax[i] * resolution[0])) - bboxs[i][1]

        return top_label_indices, top_conf, bboxs

    def predict(self, image_path):
        transformer = caffe.io.Transformer({'data': self.ssd.blobs['data'].data.shape})
        transformer.set_transpose('data', (2, 0, 1))  # row to col, (HWC -> CHW)
        transformer.set_mean('data', np.array([104, 117, 123], dtype=np.float32))
        transformer.set_raw_scale('data', 255)  # [0,1] to [0,255]
        transformer.set_channel_swap('data', (2, 1, 0))  # RGB to BGR

        image = caffe.io.load_image(image_path)  # range from 0 to 1

        self.ssd.blobs['data'].reshape(1, 3, self.nh, self.nw)
        self.ssd.blobs['data'].data[...] = transformer.preprocess('data', image)
        detections = self.ssd.forward()['detection_out']

        top_label_indices, top_conf, bboxs = self.parse_top_detection(image.shape, detections, self.conf_threshold)
        boxes = []
        for i in range(len(top_label_indices)):
            boxes.append([bboxs[i], top_conf[i], top_label_indices[i]])
        print(boxes)

        return [boxes]
