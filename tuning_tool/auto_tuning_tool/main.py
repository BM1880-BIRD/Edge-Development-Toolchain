#!/usr/bin/env python
##
## Copyright (C) Bitmain Technologies Inc.
## All Rights Reserved.
##

import argparse
import os, sys
import math, copy, shutil

os.environ['GLOG_minloglevel'] = '2'
import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format

# Path to the Calibration.so, CNet.so, caffe_net_wrapper.so
sys.path.append('../../calibration_tool/lib')
from Calibration import Calibration

from tune_utils import Tuner
from test_utils import Tester

import logging
logging.basicConfig(level=logging.WARNING)

def run_test(args):
    #  Run test.py and create report.log.
    test = Tester(args)
    test.eval_all()

def get_target_layer(ignore_layer_list):
    #  Read report.log
    with open('./report.log', 'r') as f:
        content = f.readlines()
    content = [x.strip() for x in content]

    #  Find largest diff
    pre_diff = float(content[0].split(' ', 1)[1])
    max_diff = 0
    for x in content[1:]:
        layer, diff = x.split(' ', 1)
        if float(diff) - pre_diff > max_diff and layer not in ignore_layer_list:
            target_layer = layer
            target_diff = float(diff)

            max_diff = float(diff) - pre_diff

        pre_diff = float(diff)

    return target_layer, target_diff

def run_tune(args, target_layer, target_diff):
    # Run tune.py to generate tune models
    tune = Tuner(args)
    best_thres = tune.tune_layer(target_layer, target_diff)

    # Check if the smallest is the same as origin. If so, it means this layer is already the best
    if best_thres == None:
        print('Layer {} no need to tune. Please add it to the ignore list.'.format(target_layer))
        return None, None

    # Move the smallest model
    shutil.move(os.path.join(args.output_path, target_layer.replace('/', '-') + '_thres_' + str(best_thres)), './')
    best_proto = os.path.join(target_layer.replace('/', '-') + '_thres_' + str(best_thres), "bmnet_tune_calibration_table.pb2")
    best_caffemodel = os.path.join(target_layer.replace('/', '-') + '_thres_' + str(best_thres), "bmnet_tune_int8.caffemodel")

    # Clear all other models
    for root, dirs, files in os.walk(args.output_path):
        for dir in dirs:
            shutil.rmtree(os.path.join(args.output_path, dir))
        break

    print('The best tune model: ' + best_caffemodel)
    print('The best tune proto: ' + best_proto)

    return best_caffemodel, best_proto

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # params below used for auto tuning
    parser.add_argument('--model', metavar='input-model', help='Path to fp32 caffe model')
    parser.add_argument('--proto', metavar='input-proto', help='Path to fp32 prototxt')
    parser.add_argument('--calibration_model', metavar='calibration-model', help='Path to the calibration model')
    parser.add_argument('--calibration_proto', metavar='calibration-proto', help='Path to the calibration pb2')
    parser.add_argument('--output_path', metavar='output-path', help='Output directory')
    parser.add_argument('--data_list', metavar='data-list', help='Input data path list')
    parser.add_argument('--data_limit', metavar='data-limit', help='The test data limit number')
    parser.add_argument('--image_params', metavar='image-params', help='The parameters for image preprocess')
    parser.add_argument('--ignore_layer_list', metavar='ignore-layer-list', help='Ignore these layers because they already tuned')
    args = parser.parse_args()

    caffe.set_mode_gpu()
    caffe.set_device(0)
    # Call it when you are using yolo
    # caffe.yolo()

    if not os.path.isdir(args.output_path):
	    os.mkdir(args.output_path)

    param = caffe_pb2.NetParameter()
    text_format.Merge(open(args.proto).read(), param)
    first = True
    for layer in param.layer:
        if 'data' == layer.name:
            continue
        #if 'BatchNorm' == layer.type:
        if layer.type in ['BatchNorm','Data','Input','Python','Softmax','PriorBox','Reshape', 'Flatten', 'Reshape', 'DetectionOutput']:
            continue
        if layer.name in args.ignore_layer_list:
            continue

        best_model, best_proto = run_tune(args, layer.name, sys.float_info.max)
        if first == False:
            shutil.rmtree(args.calibration_model[0:args.calibration_model.rfind('/')])
        args.calibration_model = best_model
        args.calibration_proto = best_proto
        first = False

    exit(0)