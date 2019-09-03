#!/usr/bin/env python
##
## Copyright (C) Bitmain Technologies Inc.
## All Rights Reserved.
##




import sys

#add library path
sys.path.append('./lib')

#add caffe package 
sys.path.append('..')

import caffe
from Calibration import Calibration
import argparse
import os
import time

#caffe.set_mode_gpu()
#caffe.set_device(0)

def run_calibration(args):
    calibration_info = {
        "model_name": args.model_name,
        "in_prototxt": os.path.join(args.model_path,'deploy.prototxt'),
        "in_caffemodel": os.path.join(args.model_path,'custom.caffemodel'),
        "iteration": 1,
        "enable_memory_opt": args.memory_opt,
        "enable_calibration_opt": 1,
        "histogram_bin_num": 2048,
        "math_lib_path": './lib/calibration_math.so'
    }
    print(calibration_info)

    calib = Calibration(calibration_info)
    calib.calc_tables()
    calib.export_model('{}/bmnet_{}_int8.caffemodel'.format(args.model_path, args.model_name))
    calib.export_calirabtion_pb2('{}/bmnet_{}_calibration_table'.format(args.model_path, args.model_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('model_name', metavar='model-name', help='model name')
    parser.add_argument('model_path', metavar='model-path', help='model path')

    parser.add_argument('--memory_opt', action='store_true', help='Enable memory optimization.')

    args = parser.parse_args()

    time_start = time.time()

    run_calibration(args)

    time_end = time.time()
    print('Time: %fs' % (time_end - time_start))

    sys.exit(0)

