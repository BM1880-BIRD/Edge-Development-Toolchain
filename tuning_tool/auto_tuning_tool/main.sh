python main.py \
    --proto /opt/Edge-Development-Toolchain/tuning_tool/auto_tuning_tool/Resnet50/deploy.prototxt \
    --model /opt/Edge-Development-Toolchain/tuning_tool/auto_tuning_tool/Resnet50/custom.caffemodel \
    --calibration_proto "/opt/Edge-Development-Toolchain/tuning_tool/auto_tuning_tool/Resnet50/bmnet_Resnet50_calibration_table.pb2" \
    --calibration_model "/opt/Edge-Development-Toolchain/tuning_tool/auto_tuning_tool/Resnet50/bmnet_Resnet50_int8.caffemodel" \
    --output_path /opt/Edge-Development-Toolchain/tuning_tool/auto_tuning_tool/Resnet50/ \
    --data_list /opt/Edge-Development-Toolchain/calibration_tool/Resnet50/ILSVRC2012_val/input.txt \
    --data_limit 20 \
    --image_params "image_params.json" \
    --ignore_layer_list 'data'  
