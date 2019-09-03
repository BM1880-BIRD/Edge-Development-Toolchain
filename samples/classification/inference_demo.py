import argparse
import caffe
import cv2
import numpy as np
import os

params = {}

def prepare_param(args):
    params['model'] = './{}/{}.caffemodel'.format(args.model_name, args.model_name)
    params['prototxt'] = './{}/{}.prototxt'.format(args.model_name, args.model_name)
    params['calibration_model'] = './{}/bmnet_{}_int8.caffemodel'.format(args.model_name, args.model_name)
    params['calibration_proto'] =  './{}/bmnet_{}_calibration_table.pb2'.format(args.model_name, args.model_name)
    params['imagenet_label_path'] = './imagenet_synset_to_human_label_map.txt'
    params['img_path'] = './husky.jpg'
    params['imgnet_mean'] = np.array([103.52, 116.28, 123.675], dtype=np.float32)
    params['size_h'] = 224
    params['size_w'] = 224
    params['scale'] = 255
    params['int8_flag'] = False

    params['is_color'] = True if args.model_name != "lenet" else False


def crop_center(img, crop_w, crop_h):
    h, w, _ = img.shape
    offset_h = int((h - crop_h) / 2)
    offset_w = int((w - crop_w) / 2)
    return img[offset_h:h - offset_h, offset_w:w - offset_w]


def inference_from_jpg():
    net = caffe.Net(params['prototxt'], params['model'], caffe.TEST)
    if params['int8_flag'] is True:
        net.int8_init(params['prototxt'], params['calibration_model'], params['calibration_proto'],'')

    img = caffe.io.load_image(params['img_path'], color=params['is_color'])

    h, w, _ = img.shape
    if h < w:
        off = (w - h) / 2
        img = img[:, off:off + h]
    else:
        off = (h - w) / 2
        img = img[off:off + h, :]
    img = caffe.io.resize_image(img, [params['size_h'], params['size_w']])
    print(img.shape)

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))  # row to col, (HWC -> CHW)
    if net.blobs['data'].data.shape[1] == 3:
        transformer.set_channel_swap('data', (2, 1, 0))  # RGB to BGR
        transformer.set_mean('data', params['imgnet_mean'])
    transformer.set_raw_scale('data', params['scale'])  # [0,1] to [0,255]

    net.blobs['data'].reshape(1, img.shape[2], params['size_h'], params['size_w'])
    net.blobs['data'].data[...] = transformer.preprocess('data', img)
    out = net.forward()
    prob = out['prob']
    prob = np.squeeze(prob)
    idx = np.argsort(-prob)

    label_names = np.loadtxt(params['imagenet_label_path'], str, delimiter='\t')
    for i in range(5):
        label = idx[i]
        print('%.2f - %s' % (prob[label], label_names[label]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('model_name', metavar='model-name', help='model name')
    args = parser.parse_args()

    prepare_param(args)
    inference_from_jpg()

