# BM1880-Calibration
Tools for int8 calibration &amp; finetune
```bash
├── caffe      --------------------> modified caffe library                           
│   ├── _caffe.so
│   ├── ...
├── caffe_gpu  --------------------> modified caffe library for GPU
│   ├── _caffe.so
│   └── libcaffe.so.1.0.0
├── calibration_tool  -------------> calibration tool & library
│   ├── calibration.py
│   ├── Calibration Tool Guide.pdf
│   ├── custom
│   └── lib
├── docker    ---------------------> docker support for GPU&CPU
│   ├── Dockerfile_CPU
│   └── Dockerfile_GPU
├── samples   ---------------------> some samples of our supported AI networks
│   ├── classification
│   ├── detection
│   └── super_resolution
└── tuning_tool  ------------------> tool for auto tuning our int8 caffemodel
    ├── auto_tuning_tool
    └── Auto Tuning Tool Guide.pdf
```
