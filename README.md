<table>
<tr>
<td colspan="4" align="center"><h1>BM1880 Calibration Introduction</h1>
</td>
</tr>
</table>


# Introduction

our tools for caffe model calibration   


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


