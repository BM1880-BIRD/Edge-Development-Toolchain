# Edge Development Toolchain

# Introduction

Edge calculation TPU only focus on AI model inference. Caffe FP32 model should be converted to INT8 model before deploy on TPU accelerator . Typical working flow and related tools are shown as below. 

<p align="center">
  <a href="https://github.com/BM1880-BIRD/bm1880-calibration">
    <img src="assets/working_flow.jpg" width="750px">
  </a>
</p>

We provide all tools and user guide here. 

## Snapshot of all contents

```
├── caffe
│   ├── _caffe.so
│   ├── classifier.py
│   ├── classifier.pyc
│   ├── coord_map.py
│   ├── detector.py
│   ├── detector.pyc
│   ├── draw.py
│   ├── imagenet
│   ├── __init__.py
│   ├── __init__.pyc
│   ├── io.py
│   ├── io.pyc
│   ├── libcaffe.so.1.0.0
│   ├── net_spec.py
│   ├── net_spec.pyc
│   ├── proto
│   ├── pycaffe.py
│   └── pycaffe.pyc
├── caffe_gpu
│   ├── _caffe.so
│   └── libcaffe.so.1.0.0
├── calibration_tool
│   ├── calibration.py
│   ├── Calibration Tool Guide.pdf
│   ├── custom
│   │   └── deploy.prototxt
│   └── lib
│       ├── caffe_net_wrapper.so
│       ├── calibration_math.so
│       ├── Calibration.so
│       └── CNet.so
├── docker
│   ├── Dockerfile_CPU
│   └── Dockerfile_GPU
├── samples
│   ├── classification
│   │   ├── 7.jpg
│   │   ├── alexnet
│   │   ├── calibraiton.py
│   │   ├── googlenet
│   │   ├── husky.jpg
│   │   ├── imagenet_mean.binaryproto
│   │   ├── imagenet_synset_to_human_label_map.txt
│   │   ├── inference_demo.py
│   │   ├── lenet
│   │   ├── mnist_synset_to_human_label_map.txt
│   │   ├── README
│   │   ├── resnet50
│   │   ├── squeezenet
│   │   └── vgg16
│   ├── detection
│   │   ├── calibration.py
│   │   ├── dog.jpg
│   │   ├── inference_demo.py
│   │   ├── input.txt
│   │   ├── labelmap_coco.prototxt
│   │   ├── labelmap_voc.prototxt
│   │   ├── README.md
│   │   ├── ssd300
│   │   ├── ssd512
│   │   ├── utils.py
│   │   └── yolov3
│   └── super_resolution
│       └── espcn
└── tuning_tool
    ├── auto_tuning_tool
    │   ├── evaluation_utils.py
    │   ├── main.py
    │   ├── main.sh
    │   ├── test.py
    │   ├── test.sh
    │   ├── test_utils.py
    │   ├── tune.py
    │   ├── tune.sh
    │   └── tune_utils.py
    └── Auto Tuning Tool Guide.pdf
```

## Guide 


### caffe

This folder include our customized python caffe which support INT8 mode data and model. Calibration Tool and  tuning tool  will use it.  What's more you can also use it for offline model test. You can refer to sample part for reference. 

This part caffe library is provided for CPU. 

### caffe_gpu

This part include GPU caffe library. If you use GPU caffe , you need to replace _caffe.so and libcaffe.so.1.0.0 in caffe folder. 

### calibration_tool

You can use Calibration tool to do FP32 caffe model quantization to convert it to INT8 model. There is  user guide  "Calibration Tool Guide.pdf" to explain detail. 

### docker

We provide both CPU and GPU docker files for you to set up docker enviroment to use these tools. 

### tuning_tool

You can fine tune your int8 model if the accuracy is not satisfied. There is  user guide  "Auto Tuning Tool Guide.pdf" to explain detail. 

### samples

We provide a lot of frequently used network calibration and offline test samples here. 

