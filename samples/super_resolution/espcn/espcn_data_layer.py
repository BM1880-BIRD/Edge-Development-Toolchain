#!/usr/bin/env python
##
## Copyright (C) Bitmain Technologies Inc.
## All Rights Reserved.
##

import os, sys
import caffe
import yaml
import cv2
import numpy as np

class DataLayer(caffe.Layer):

    def setup(self, bottom, top):
        """Setup the DataLayer."""
        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str)
        list_name = layer_params['image_list']
        self.input_hw = layer_params['hw']
        self.fp = open(list_name)
        # data blob: holds a batch of N images, each with 3 channels
        top[0].reshape(1, 1, self.input_hw, self.input_hw)

    def reset(self):
        self.fp.seek(0)

    def forward(self, bottom, top):
        pic_name = self.fp.readline()
        if pic_name == "" :
            self.fp.seek(0)
            pic_name = self.fp.readline()
        pic_name = pic_name.strip('\n')
        print(pic_name)
        
        lr_image_data = cv2.imread(pic_name)
        inp_dim = (self.input_hw, self.input_hw)
        lr_image_data = cv2.resize(lr_image_data, inp_dim, interpolation = cv2.INTER_CUBIC)
	lr_image_ycbcr_data = cv2.cvtColor(lr_image_data, cv2.COLOR_BGR2YCrCb)
        lr_image_y_data = lr_image_ycbcr_data[:, :, 0:1]
        lr_image_cb_data = lr_image_ycbcr_data[:, :, 1:2]
        lr_image_cr_data = lr_image_ycbcr_data[:, :, 2:3]
        lr_image_batch = np.zeros((1,) + lr_image_y_data.shape)
        transformed_image = lr_image_y_data.transpose([2,0,1]) / 255.0

        # Reshape net's input blobs
        top[0].reshape(1, 1, self.input_hw, self.input_hw)
        # Copy data into net's input blobs
        top[0].data[...] = transformed_image

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

