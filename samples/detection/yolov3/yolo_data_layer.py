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
from utils import letterbox_image

class YoloDataLayer(caffe.Layer):

    def setup(self, bottom, top):
        """Setup the DataLayer."""
        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str)
        list_name = layer_params['image_list']
        self.input_hw = layer_params['hw']
        self.fp = open(list_name)
        # data blob: holds a batch of N images, each with 3 channels
        top[0].reshape(1, 3, self.input_hw, self.input_hw)

    def forward(self, bottom, top):
        pic_name = self.fp.readline()
        if pic_name == "" :
            self.fp.seek(0)
            pic_name = self.fp.readline()
        pic_name = pic_name.strip('\n')
        print(pic_name)

        img = cv2.imread(pic_name)
        transformed_image = letterbox_image(img, self.input_hw, self.input_hw)

        # Reshape net's input blobs
        top[0].reshape(1, 3, self.input_hw, self.input_hw)
        # Copy data into net's input blobs
        top[0].data[...] = transformed_image

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

