# Calibration and test data set link

## COCO
https://pjreddie.com/media/files/val2014.zip

## VOC
cuda2:/data/dataset/VOCdevkit/VOC0712/lmdb/VOC0712_test_lmdb_512

Please put the datasets in the parent folder.

# Calibration

```
# python calibration yolov3 --memory_opt
```

# Demo

##

1.Need to check the image size and int8 flag.
2.'enable_calibration_opt' should disable when calibrate ssd.

```
# python inference_demo.py yolov3
```
