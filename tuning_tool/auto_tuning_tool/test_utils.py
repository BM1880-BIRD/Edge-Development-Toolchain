import cv2
import numpy as np
import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format
import sys, os, copy, math
import lmdb
import json
import copy

import evaluation_utils as eva_utils

def parse_data_list(data_list):
    with open(data_list) as f:
        img_list = f.readlines()
        img_list = [x.strip() for x in img_list] 

    return img_list

def rescale_image(image, size, pad):
    if size is None:  # original resolution
        return image
    if image.shape[0] == size[0] and image.shape[1] == size[1]:
        return image

    if isinstance(size[0], int) and isinstance(size[1], int):
        if pad:
            ratio = max(float(image.shape[1])/float(size[1]), float(image.shape[0])/float(size[0]))
            if ratio != 1.:
                image = cv2.resize(image, (int(image.shape[1]/ratio), int(image.shape[0]/ratio)),
                                   interpolation=cv2.INTER_LINEAR)

            pad_left = int((size[1]-image.shape[1])/2)
            pad_top = int((size[0]-image.shape[0])/2)
            pad_right = size[1]-image.shape[1]-pad_left
            pad_bottom = size[0]-image.shape[0]-pad_top
            rescale_param = (ratio, ratio, pad_top, pad_left)
            if pad_left != 0 or pad_top != 0 or pad_right !=0 or pad_bottom != 0:
                image = cv2.copyMakeBorder(image, top=pad_top, left=pad_left, bottom=pad_bottom,
                                           right=pad_right, borderType=cv2.BORDER_CONSTANT, value=0)
        else:
            rescale_param = (image.shape[1]/size[1], image.shape[0]/size[0], 0, 0)
            image = cv2.resize(image, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)
    else:
        raise TypeError('not valid ', size)

    return image

class Tester(object):
    def __init__(self, args):
        self.proto = args.proto
        self.model = args.model
        self.calibration_proto = args.calibration_proto
        self.calibration_model = args.calibration_model

        self.img_list = parse_data_list(args.data_list)
        self.limit = int(args.data_limit)

        self.param = caffe_pb2.NetParameter()
        text_format.Merge(open(args.proto).read(), self.param)

        with open(args.image_params) as json_file:
            self.image_params = json.load(json_file)

        self.out32 = self.net32_inference()

    def set_calibration_model(self, model):
        self.calibration_model = model

    def set_calibration_proto(self, proto):
        self.calibration_proto = proto

    def prepare_image(self, image):
        size = [256, 256] if 'size' not in self.image_params else self.image_params['size']
        padding = False if 'padding' not in self.image_params else self.image_params['padding']
        mirror = -2 if 'mirror' not in self.image_params else int(self.image_params['mirror'])
        color_format = 'rgb' if 'color_format' not in self.image_params else self.image_params['color_format']
        r_mean = 0. if 'r_mean' not in self.image_params else float(self.image_params['r_mean'])
        g_mean = 0. if 'g_mean' not in self.image_params else float(self.image_params['g_mean'])
        b_mean = 0. if 'b_mean' not in self.image_params else float(self.image_params['b_mean'])
        scale = 1. if 'scale' not in self.image_params else float(self.image_params['scale'])

        resized_image = rescale_image(image, size, padding)

        if mirror in [-1, 0, 1]:
            flip_image = cv2.flip(resized_image, mirror)
            resized_image = flip_image

        if color_format == 'rgb':
            resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
            resized_image = resized_image - [r_mean, g_mean, b_mean]
        else:
            resized_image = resized_image - [b_mean, g_mean, r_mean]

        resized_image *= scale

        transformed_image = resized_image.transpose([2,0,1])

        return transformed_image

    def eval_all(self):
        fp = open('./report.log', "w", buffering=1)
        layer_str = ''

        for layer in self.param.layer:
            layer_str += str(layer.name)

            # BatchNorm and Scale layer must open int8 together
            if layer.type == 'BatchNorm':
                layer_str += ','
                continue

            layer_dist = self.net8_inference(layer_str, self.out32)

            fp.write(layer.name + " ")
            fp.write(str(layer_dist) + "\n")
            layer_str += ','

        fp.close()
        print('Evaluate all layer done!!')

    def eval_one(self, target_layer):
        layer_str = ''
        for layer in self.param.layer:
            if layer.type not in ['Python','Softmax','Flatten', 'Reshape']:
                layer_str+=str(layer.name)
                if (str(layer.name) == target_layer):
                    break
                layer_str+=","

        layer_dist = self.net8_inference(layer_str, self.out32)

        return layer_dist

    def net32_inference(self):
        net_32 = caffe.Net(self.proto, self.model, caffe.TEST)
        idx = 0
        out32 = []

        for impath in self.img_list:
            if idx >= self.limit:
                break
            idx+=1

            ori_image = cv2.imread(impath)
            image = self.prepare_image(ori_image)

            net_32.blobs['data'].data[...] = image
            out32.append(copy.deepcopy(net_32.forward()))

        return out32

    def net8_inference(self, layer_str, out32):
        net_8 = caffe.Net(self.proto, self.model, caffe.TEST)
        net_8.int8_init(self.proto,
                        self.calibration_model,
                        self.calibration_proto,
                        layer_str)

        layer_dist = 0
        idx = 0

        for impath in self.img_list:
            if idx >= self.limit:
                break

            ori_image = cv2.imread(impath)
            image = self.prepare_image(ori_image)

            net_8.blobs['data'].data[...] = image
            out8 = net_8.forward()

            for i in range(len(net_8.outputs)):
                layer_dist += eva_utils.L2_diff(out32[idx][net_8.outputs[i]], out8[net_8.outputs[i]])

            idx+=1

        return layer_dist / idx