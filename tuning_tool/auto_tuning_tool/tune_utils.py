import cv2
import numpy as np
import caffe
from caffe.proto.bmnet import common_calibration_pb2
from google.protobuf import text_format
import sys, os, copy, math, shutil

from Calibration import Calibration
from test_utils import Tester

def parse_calibration_proto(proto):
    with open(proto, 'rb') as f:
        cali_param = common_calibration_pb2.NetCalibrationParameter()
        cali_param.ParseFromString(f.read())

    thresholds = {}
    for layer in cali_param.layer:
        name = layer.name
        thresholds[name] = float(layer.threshold_y[0])

    return thresholds

def run_calibration(in_proto, in_caffemodel, out_table, out_caffemodel, thresholds=None):
    calibration_info = {
        "model_name": 'tune',
        "in_prototxt": in_proto,
        "in_caffemodel": in_caffemodel,
        "out_prototxt": './deploy_out.prototxt',
        "iteration": 1,
        "enable_memory_opt": 0,
        "math_lib_path": '../../calibration_tool/lib/calibration_math.so'
    }
    print(calibration_info)

    calib = Calibration(calibration_info)
    calib.tune_tables(thresholds)
    calib.export_model(out_caffemodel)
    calib.export_calirabtion_pb2(out_table)

class Tuner(object):
    def __init__(self, args):
        self.proto = args.proto
        self.model = args.model
        self.calibration_proto = args.calibration_proto
        self.calibration_model = args.calibration_model

        self.output_path = args.output_path
        self.out_table = os.path.join(self.output_path, "bmnet_tune_calibration_table")
        self.out_proto = os.path.join(self.output_path, "bmnet_tune_calibration_table.prototxt")
        self.out_pb = os.path.join(self.output_path, "bmnet_tune_calibration_table.pb2")
        self.out_caffemodel = os.path.join(self.output_path, "bmnet_tune_int8.caffemodel")

        self.thresholds = parse_calibration_proto(self.calibration_proto)

        self.enlarge_factor = 1.01
        self.reduce_factor = 0.99

        self.best_threshold = 0
        self.best_diff = 0

        self.test = Tester(args)

    def tune_layer(self, target_layer, target_diff):
        ori_diff = target_diff
        ori_thres = self.thresholds.get(target_layer)
        print('ori_thres={}, ori_diff={}'.format(ori_thres, ori_diff))

        self.best_threshold = ori_thres
        self.best_diff = ori_diff

        self.get_layer_best_threshold(ori_thres, ori_diff, target_layer, self.enlarge_factor)
        self.get_layer_best_threshold(ori_thres, ori_diff, target_layer, self.reduce_factor)

        print('tuning end, layer: {},  best diff with tune: {}/{}, threshold: {}/{}'.format(
            target_layer, self.best_diff, ori_diff, self.best_threshold, ori_thres))

        return self.best_threshold

    def get_layer_best_threshold(self, ori_thres, ori_diff, tune_layer, factor):
        count = 0
        fail_count = 0
        pre_diff = ori_diff

        tune_thresholds = copy.deepcopy(self.thresholds)

        while fail_count < 3:
            tune_thres = ori_thres * math.pow(factor, count)
            tune_thresholds[tune_layer] = tune_thres

            print('start tuning: {}, layer: {}, tuning threshold: {}'.format(count + 1, tune_layer, tune_thres))

            run_calibration(self.proto, self.model, self.out_table, self.out_caffemodel, tune_thresholds)

            self.test.set_calibration_model(self.out_caffemodel)
            self.test.set_calibration_proto(self.out_pb)
            diff = self.test.eval_one(tune_layer)

            print('end tuning: {}, layer: {}, tuning diff: {}'.format(count + 1, tune_layer, diff))

            if self.best_diff > diff:
                #  Remove previous saved best model/proto
                if os.path.isdir(os.path.join(self.output_path, "{}_thres_{}".format(tune_layer.replace('/', '-'), self.best_threshold))):
                    shutil.rmtree(os.path.join(self.output_path, "{}_thres_{}".format(tune_layer.replace('/', '-'), self.best_threshold)))

                thres_fold = os.path.join(self.output_path, "{}_thres_{}".format(tune_layer.replace('/', '-'), tune_thres))
                try:
                    if not os.path.isdir(thres_fold):
                        os.mkdir(thres_fold)

                    shutil.copy(self.out_table, thres_fold)
                    shutil.copy(self.out_proto, thres_fold)
                    shutil.copy(self.out_pb, thres_fold)
                    shutil.copy(self.out_caffemodel, thres_fold)
                except (OSError, IOError) as e:
                    print(e)

                self.best_diff = diff
                self.best_threshold = tune_thres
                fail_count = 0
            else:
                if pre_diff <= diff:
                    fail_count += 1

            pre_diff = diff
            count += 1
