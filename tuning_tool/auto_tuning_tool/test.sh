python test.py \
	--proto /home/terry/face-alignment/fan_net_official.prototxt \
	--model /home/terry/face-alignment/fan_net_official.caffemodel \
	--calibration_proto /home/terry/calibration/headpose/bmnet_fan_calibration_table.1x10.pb2 \
	--calibration_model /home/terry/calibration/headpose/bmnet_fan_int8.1x10.caffemodel \
	--data_list /home/terry/algorithm_test/face/fan_net/list.txt \
  	--data_limit 10 \
	--image_params "image_params.json" \
	--int8_layer ''
