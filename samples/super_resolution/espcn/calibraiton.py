#!/usr/bin/env python
##
## Copyright (C) Bitmain Technologies Inc.
## All Rights Reserved.
##

import argparse
import caffe
import collections
import os
import sys
import time

sys.path.append('../../calibration_tool/lib')
from Calibration import Calibration

#caffe.set_mode_gpu()
#caffe.set_device(0)

def run_calibration(arg):
    calibration_info = {
        "model_name": arg.model_name,
        "in_prototxt": os.path.join(arg.model_path, 'deploy_2x.prototxt'),
        "in_caffemodel": os.path.join(arg.model_path, 'espcn.caffemodel'),
        "iteration": 100,
        "enable_memory_opt": arg.memory_opt,
        "histogram_bin_num": 2048,
        "math_lib_path": "../../calibration_tool/build/calibration_math.so"
    }
    print(calibration_info)

    calib = Calibration(calibration_info)
    calib.calc_tables()
    calib.export_model(os.path.join(arg.model_path, 'bmnet_espcn_int8.caffemodel'))
    calib.export_calirabtion_pb2(os.path.join(arg.model_path, 'bmnet_espcn_calibration_table'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('model_name', metavar='model-name', help='model name')
    parser.add_argument('model_path', metavar='model-path', help='model path')
    parser.add_argument('--memory_opt', action='store_true', help='enable memory optimization')

    arg = parser.parse_args()
    assert os.path.exists(arg.model_path), "path {} does not exist".format(arg.model_path)

    time_start = time.time()

    run_calibration(arg)

    time_end = time.time()
    print('Time: %fs' % (time_end - time_start))

    sys.exit(0)
