# ------------------------------------------------------------------------------------------------
# This file is a modified version of https://github.com/rbgirshick/py-faster-rcnn by Ross Girshick
# Modified by Mahyar Najibi
# ------------------------------------------------------------------------------------------------
import numpy as np
import os

import caffe
import yaml
import cv2

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

class DataLayer(caffe.Layer):

    def setup(self, bottom, top):
        """Setup the DataLayer."""
        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str)
        print(layer_params)

        list_name = layer_params['data_list']
        self.input_h = layer_params['h']
        self.input_w = layer_params['w']

        self.color_format = 'rgb'
        self.r_mean = 0
        self.g_mean = 0
        self.b_mean = 0
        self.scale = 1
        self.mirror = 0
        self.transpose = [2,0,1]
        self.padding = False

        if 'color_format' in layer_params:
            self.color_format = layer_params['color_format']
        if 'r_mean' in layer_params:
            self.r_mean = layer_params['r_mean']
        if 'g_mean' in layer_params:
            self.g_mean = layer_params['g_mean']
        if 'b_mean' in layer_params:
            self.b_mean = layer_params['b_mean']
        if 'scale' in layer_params:
            self.scale = layer_params['scale']
        if 'mirror' in layer_params:
            self.mirror = layer_params['mirror'] - 2
        if 'transpose' in layer_params:
            self.transpose = layer_params['transpose']
        if 'padding' in layer_params:
            self.padding = layer_params['padding']

        self.debug = None
        if 'debug' in layer_params:
            self.debug = layer_params['debug']
            if not os.path.exists('./debug'):
                os.makedirs('./debug')

        self.fp = open(list_name)
        # data blob: holds a batch of N images, each with 3 channels
        top[0].reshape(1, 3, self.input_h, self.input_w)

    def reset(self):
        self.fp.seek(0)

    def forward(self, bottom, top):
        pic_name = self.fp.readline()
        if pic_name == "" :
            self.fp.seek(0)
            pic_name = self.fp.readline()
        pic_name = pic_name.strip('\n')
        print(pic_name)

        img = cv2.imread(pic_name)
        img = img.astype(np.float32)
        inp_dim = (self.input_h, self.input_w)
        resized_image = rescale_image(img, inp_dim, self.padding)
        # resized_image = cv2.resize(img, inp_dim, interpolation = cv2.INTER_LINEAR)

        if self.mirror != 0:
            flip_image = cv2.flip(resized_image, self.mirror)
            resized_image = flip_image

        if self.color_format == 'rgb':
            resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
            resized_image = resized_image - [self.r_mean, self.g_mean, self.b_mean]
        else:
            resized_image = resized_image - [self.b_mean, self.g_mean, self.r_mean]

        resized_image *= self.scale

        transformed_image = resized_image.transpose(self.transpose)
        # Reshape net's input blobs
        top[0].reshape(1, 3, self.input_h, self.input_w)
        # Copy data into net's input blobs
        top[0].data[...] = transformed_image

        if self.debug == 'npy':
            np.save('./debug/' + pic_name[pic_name.rfind('/')+1:], np.asarray(resized_image))
        elif self.debug == 'image':
            cv2.imwrite('./debug/' + pic_name[pic_name.rfind('/')+1:], resized_image)

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

