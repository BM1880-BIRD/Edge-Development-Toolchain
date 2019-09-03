python main.py \
	--proto /home/terry/calibration/zkzy/yolov3.prototxt \
	--model /home/terry/calibration/zkzy/custom.caffemodel \
        --calibration_proto "/home/terry/calibration/zkzy/bmnet_zkzy_calibration_table.1x10.pb2" \
        --calibration_model "/home/terry/calibration/zkzy/bmnet_zkzy_int8.1x10.caffemodel" \
	--output_path ./result_tune \
        --data_list /home/terry/calibration/zkzy/input.txt \
	--data_limit 20 \
	--image_params "image_params.json" \
	--ignore_layer_list 'data'
