<table>
<tr>
<td colspan="4" align="center"><h1>BM1880 Calibration Introduction</h1>
</td>
</tr>
</table>


# Introduction

this is mainly for calibration fp32 caffemodel in


## Organization

we supply modified caffe/caffe_gpu library for special layers and int8 support.  
the calibration_tools/tuning_tool are for calibration and auto finetune caffemodel. you can also see detail guide in each folder.  
we also supply docker for convenience  

```
.
├── caffe
│   ├── _caffe.so
│   ├── classifier.py
│   ├── classifier.pyc
│   ├── coord_map.py
│   ├── detector.py
│   ├── detector.pyc
│   ├── draw.py
│   ├── imagenet
│   │   └── ilsvrc_2012_mean.npy
│   ├── __init__.py
│   ├── __init__.pyc
│   ├── io.py
│   ├── io.pyc
│   ├── libcaffe.so.1.0.0
│   ├── net_spec.py
│   ├── net_spec.pyc
│   ├── proto
│   │   ├── bmnet
│   │   │   ├── common_calibration_pb2.py
│   │   │   ├── common_calibration_pb2.pyc
│   │   │   ├── __init__.py
│   │   │   └── __init__.pyc
│   │   ├── caffe_pb2.py
│   │   ├── caffe_pb2.pyc
│   │   ├── __init__.py
│   │   └── __init__.pyc
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
│   │   │   ├── alexnet.prototxt
│   │   │   ├── bmnet_alexnet.prototxt
│   │   │   ├── deploy.prototxt
│   │   │   └── README
│   │   ├── calibraiton.py
│   │   ├── googlenet
│   │   │   ├── deploy.prototxt
│   │   │   ├── googlenet.prototxt
│   │   │   └── README
│   │   ├── husky.jpg
│   │   ├── imagenet_mean.binaryproto
│   │   ├── imagenet_synset_to_human_label_map.txt
│   │   ├── inference_demo.py
│   │   ├── lenet
│   │   │   ├── deploy.prototxt
│   │   │   ├── lenet.prototxt
│   │   │   └── README
│   │   ├── mnist_synset_to_human_label_map.txt
│   │   ├── README
│   │   ├── resnet50
│   │   │   ├── bmnet_resnet50.prototxt
│   │   │   ├── deploy.prototxt
│   │   │   ├── README
│   │   │   └── resnet50.prototxt
│   │   ├── squeezenet
│   │   │   ├── deploy.prototxt
│   │   │   ├── README
│   │   │   └── squeezenet.prototxt
│   │   └── vgg16
│   │       ├── deploy.prototxt
│   │       ├── README
│   │       └── vgg16.prototxt
│   ├── detection
│   │   ├── calibration.py
│   │   ├── dog.jpg
│   │   ├── inference_demo.py
│   │   ├── input.txt
│   │   ├── labelmap_coco.prototxt
│   │   ├── labelmap_voc.prototxt
│   │   ├── README.md
│   │   ├── ssd300
│   │   │   ├── deploy.prototxt
│   │   │   ├── general_data_layer.py
│   │   │   ├── __init__.py
│   │   │   ├── ssd300.prototxt
│   │   │   ├── ssd.py
│   │   │   └── ssd.pyc
│   │   ├── ssd512
│   │   │   ├── deploy_modify.prototxt
│   │   │   ├── deploy.prototxt
│   │   │   ├── README
│   │   │   └── ssd512.prototxt
│   │   ├── utils.py
│   │   └── yolov3
│   │       ├── deploy.prototxt
│   │       ├── __init__.py
│   │       ├── README
│   │       ├── yolo_data_layer.py
│   │       ├── yolo.py
│   │       └── yolov3.prototxt
│   └── super_resolution
│       └── espcn
│           ├── calibraiton.py
│           ├── deploy_2x.prototxt
│           ├── espcn_2x.prototxt
│           ├── espcn_data_layer.py
│           ├── input.txt
│           ├── lenna.bmp
│           ├── README
│           └── test_espcn.py
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


