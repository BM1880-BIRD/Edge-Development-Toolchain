#!/usr/bin/env python
##
## Copyright (C) Bitmain Technologies Inc.
## All Rights Reserved.
##

import argparse
import caffe
import os
import sys
import time

sys.path.append('../../calibration_tool/lib')
from Calibration import Calibration

#caffe.set_mode_gpu()
#caffe.set_device(0)

def run_calibration(args):
    calibration_info = {
        "model_name": args.model_name,
        "in_prototxt": './{}/deploy.prototxt'.format(args.model_name),
        "in_caffemodel": './{}/{}.caffemodel'.format(args.model_name, args.model_name),
        "iteration": 20,
        "enable_memory_opt": args.memory_opt,
        "enable_calibration_opt": 1,
        "histogram_bin_num": 2048,
        "math_lib_path": '../../calibration_tool/lib/calibration_math.so'
    }
    print(calibration_info)

    calib = Calibration(calibration_info)
    calib.calc_tables()
    calib.export_model('./{}/bmnet_{}_int8.caffemodel'.format(args.model_name, args.model_name))
    calib.export_calirabtion_pb2('./{}/bmnet_{}_calibration_table'.format(args.model_name, args.model_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('model_name', metavar='model-name', help='model name')
    parser.add_argument('--memory_opt', action='store_true', help='Enable memory optimization.')

    args = parser.parse_args()

    time_start = time.time()

    run_calibration(args)

    time_end = time.time()
    print('Time: %fs' % (time_end - time_start))

    sys.exit(0)

