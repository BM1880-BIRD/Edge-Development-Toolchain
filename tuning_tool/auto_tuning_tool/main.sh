python main.py \
    --proto /home/chjtest/compiler/Edge-Development-Toolchain/tuning_tool/auto_tuning_tool/Resnet50/deploy.prototxt \
    --model /home/chjtest/compiler/Edge-Development-Toolchain/tuning_tool/auto_tuning_tool/Resnet50/custom.caffemodel \
    --calibration_proto "/home/chjtest/compiler/Edge-Development-Toolchain/tuning_tool/auto_tuning_tool/Resnet50/bmnet_Resnet50_calibration_table.pb2" \
    --calibration_model "/home/chjtest/compiler/Edge-Development-Toolchain/tuning_tool/auto_tuning_tool/Resnet50/bmnet_Resnet50_int8.caffemodel" \
    --output_path /home/chjtest/compiler/Edge-Development-Toolchain/tuning_tool/auto_tuning_tool/Resnet50/ \
    --data_list /home/chjtest/compiler/Edge-Development-Toolchain/tuning_tool/auto_tuning_tool/Resnet50/ILSVRC2012_val/input.txt \
    --data_limit 20 \
    --image_params "image_params.json" \
    --ignore_layer_list 'data'  
