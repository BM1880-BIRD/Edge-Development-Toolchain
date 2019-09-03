import numpy as np
import caffe
import sys
import argparse
from test_utils import Tester

def eval_net(args):
    test = Tester(args)

    if args.int8_layer == '':
        test.eval_all()
    else:
        diff = test.eval_one(args.int8_layer)
        print('Difference from first layer to {} layer is {}'.format(args.int8_layer, diff))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', metavar='input-model', help='Path to fp32 caffe model')
    parser.add_argument('--proto', metavar='input-proto', help='Path to fp32 prototxt')
    parser.add_argument('--calibration_model', metavar='calibration-model', help='Path to the calibration model')
    parser.add_argument('--calibration_proto', metavar='calibration-proto', help='Path to the calibration pb2')
    parser.add_argument('--data_list', metavar='data-list', help='Input data path list')
    parser.add_argument('--data_limit', metavar='data-limit', help='The test data limit number')
    parser.add_argument('--image_params', metavar='image-[arams', help='The parameters for image preprocess')
    parser.add_argument('--int8_layer', metavar='int8-layer', help='The layer name to inference in int8 mode. Empty means all')
    args = parser.parse_args()

    caffe.set_mode_gpu()
    caffe.set_device(0)

    eval_net(args)
