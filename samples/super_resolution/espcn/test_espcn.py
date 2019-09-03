import numpy as np
import json
import pdb
import caffe, cv2

# caffe.set_mode_gpu()
# caffe.set_device(0)

params = {}
params['ratio'] = 2
params['edge'] = 8
params['prototxt'] = './espcn_2x.prototxt'
params['caffemodel'] = './espcn_2x.caffemodel'
params['calibration_proto'] = './bmnet_espcn_calibration_table.pb2'
params['calibration_model'] = './bmnet_espcn_int8.caffemodel'
params['image'] = './lenna.bmp'
params['output_prefix'] = './result'
params['size_h'] = 512
params['size_w'] = 512
params['int8_flag'] = False

def shuffle(input_image, ratio):
    shape = input_image.shape
    height = int(shape[0]) * ratio
    width = int(shape[1]) * ratio
    channels = int(shape[2]) / ratio / ratio
    shuffled = np.zeros((height, width, channels), dtype=np.uint8)
    for i in range(0, height):
        for j in range(0, width):
            for k in range(0, channels):
                shuffled[i,j,k] = input_image[i / ratio, j / ratio, k * ratio * ratio + (i % ratio) * ratio + (j % ratio)]
    return shuffled


def generate():
    lr_image_data = cv2.imread(params['image'])
    lr_image_data = cv2.resize(lr_image_data, (512, 512), interpolation = cv2.INTER_CUBIC)
    lr_image_ycbcr_data = cv2.cvtColor(lr_image_data, cv2.COLOR_BGR2YCrCb)

    lr_image_y_data = lr_image_ycbcr_data[:, :, 0:1]

    net = caffe.Net(params['prototxt'], params['caffemodel'],  caffe.TEST)
    if params['int8_flag'] is True:
        net.int8_init(params['prototxt'], params['calibration_model'], params['calibration_proto'],'')

    net.blobs['data'].data[...] = lr_image_y_data.transpose([2,0,1]) / 255.0
    lmp_out = net.forward()['Conv2D_2'] * 255
    lmp_out = np.clip(lmp_out, 0, 255)


    sr_image_y_data = lmp_out.transpose([0,2,3,1])
    sr_image_y_data = shuffle(sr_image_y_data[0], params['ratio'])
    sr_image_ycbcr_data = cv2.resize(lr_image_ycbcr_data,
                                     tuple(params['ratio'] * np.array(lr_image_data.shape[0:2])), interpolation = cv2.INTER_CUBIC)

    edge = params['edge'] * params['ratio'] / 2

    sr_image_ycbcr_data = np.concatenate((sr_image_y_data, sr_image_ycbcr_data[edge:-edge,edge:-edge,1:3]), axis=2)
    sr_image_data = cv2.cvtColor(sr_image_ycbcr_data, cv2.COLOR_YCrCb2BGR)

    cv2.imwrite(params['output_prefix'] + '.png', sr_image_data)

    sr_image_bicubic_data = cv2.resize(lr_image_data,
                                       tuple(params['ratio'] * np.array(lr_image_data.shape[0:2])), interpolation = cv2.INTER_CUBIC)
    cv2.imwrite(params['output_prefix'] + '_bicubic.png', sr_image_bicubic_data)

if __name__ == '__main__':
    generate()
