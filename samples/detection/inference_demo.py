import argparse
import cv2
from utils import draw, letterbox_image
from yolov3.yolo import YOLO
from ssd300.ssd import SSD
import caffe
import os

params = {}

def prepare_param(args):
    params['model'] = './{}/{}.caffemodel'.format(args.model_name, args.model_name)
    params['prototxt'] = './{}/{}.prototxt'.format(args.model_name, args.model_name)
    params['calibration_model'] = './{}/bmnet_{}_int8.caffemodel'.format(args.model_name, args.model_name)
    params['calibration_proto'] =  './{}/bmnet_{}_calibration_table.pb2'.format(args.model_name, args.model_name)
    params['img_path'] = './dog.jpg'
    params['size_h'] = 512
    params['size_w'] = 512
    params['int8_flag'] = False
    params['output_prefix'] = './{}/'.format(args.model_name)

    if args.model_name == 'yolov3':
        params['obj_threshold'] = 0.3
        params['nms_threshold'] = 0.5
        params['num_of_class'] = 80
        params['anchors'] = [[116,90, 156,198, 373,326], [30,61, 62,45, 59,119], [10,13, 16,30, 33,23]]
        params['labelmap_file'] = './labelmap_coco.prototxt'

        detector = YOLO(params['obj_threshold'], params['nms_threshold'], params['size_w'], params['size_h'],
                params['num_of_class'], proto_path=params['prototxt'], model_path=params['model'],
                cali_proto=params['calibration_proto'], cali_model=params['calibration_model'], int8_flag=params['int8_flag'],
                anchors=params['anchors'])
    elif args.model_name in ['ssd512', 'ssd300']:
        params['conf_threshold'] = 0.6
        params['labelmap_file'] = './labelmap_voc.prototxt'
        detector = SSD(params['conf_threshold'], params['size_w'], params['size_h'], proto_path=params['prototxt'], model_path=params['model'],
                cali_proto=params['calibration_proto'], cali_model=params['calibration_model'], int8_flag=params['int8_flag'])
    else:
        print('Model {} is not supported.'.format(args.model_name))
        exit(0)

    return detector


def inference_from_jpg(detector):
    detections = detector.predict(params['img_path'])
    if detections is not None:
        img = cv2.imread(params['img_path'])
        img = draw(img, detections[0], params['labelmap_file'])
        cv2.imwrite(os.path.join(params['output_prefix'], 'result.jpg'), img)
        #cv2.imshow('result: ', img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('model_name', metavar='model-name', help='model name')
    args = parser.parse_args()

    detector = prepare_param(args)
    inference_from_jpg(detector)
