# Edge Development Toolchain

# Introduction

Edge calculation TPU only focus on AI model inference. Caffe FP32 model should be converted to INT8 model before deploy on TPU accelerator . Typical working flow and related tools are shown as below. 

<p align="center">
  <a href="https://github.com/BM1880-BIRD/bm1880-calibration">
    <img src="assets/working_flow.jpg" width="750px">
  </a>
</p>

We provide all tools and user guide here. 

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

## Step by Step  

#### 1. Environment　setup

We **strongly recommend** that you have one Nvidia GPU PC and setup GPU docker environment, which will fully speed up development process. 

* ##### Host PC 
ubuntu 18.04 or 16.04

python2.7

docker installed

nvidia-docker installed(if you want to setup GPU docker)

* ##### Setup GPU docker envirement（**Recommended**)

Currently we only provide pre-built libraries based on **CUDA 9.0**.Please make sure your host PC GPU driver is compatible with it .

First please overwrite the prebuilt shared library files in the caffe_gpu folder
to caffe folder.

Then run docker build and run as below.

sudo nvidia-docker build -t bmcalibration:2.0_gpu -f docker/Dockerfile_GPU .

sudo nvidia-docker run -v /workspace:/workspace -it bmcalibration:2.0_gpu

After that you have already completed GPU docker env. You must call below  apis  to enable GPU  Caffe.

```
caffe.set_mode_gpu()
caffe.set_device(device_id)    # device_id is  the GPU id on your machine.
```

* ##### Setup CPU docker envirement

Run docker build and run as below.

cd Edge-Development-Toolchain
sudo docker build -t bmcalibration:2.0_cpu -f docker/Dockerfile_CPU .
sudo docker run -v /workspace:/workspace -it bmcalibration:2.0_cpu

After that you have already completed CPU docker env.

#### 2. Do quantization with calibration tool

With your FP32 caffe  model (model &weight), you can use calibration tool to do FP32 quantization to get INT8 model which can be deployed in our SOC chip.  

Please refer to this [`guide`](https://github.com/BM1880-BIRD/Edge-Development-Toolchain/blob/master/calibration_tool/Calibration-Tool-Guide.pdf) to use calibration tool.

#### 3. Do offline accuracy test

You can use our customized caffe which support INT8 model to do accuracy offline test.
Currently we provide some inference [`samples`](https://github.com/BM1880-BIRD/Edge-Development-Toolchain/tree/master/samples/) 
 for your reference(including classification/detection/super_resolution).
 
 #### 4. Do model tuning
 
 If you　get unacceptable INT8 accuracy after Step 3, you should demply model tuning to finetune INT8 model. 
 
 Please refer to this [`guide`](https://github.com/BM1880-BIRD/Edge-Development-Toolchain/blob/master/tuning_tool/Auto-Tuning-Tool-Guide.pdf) to use tuning tool.
 
  #### 5. Deploy your model on platform 
  
After all above steps,  you can convert INT8 model to bmodel file and then deploy on real platform for online test. 

Please use bmbuilder tool  to convert INT8 model to bmodel. Please refer to this guide. 
 
 




