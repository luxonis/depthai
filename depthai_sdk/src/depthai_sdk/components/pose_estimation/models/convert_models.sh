# This script has to be run from the docker container started by ./docker_tflite2tensorflow.sh

source /opt/intel/openvino_2021/bin/setupvars.sh

replace () { # Replace in file $1 occurences of the string $2 by $3
        sed "s/${2}/${3}/" $1 > tmpf
        mv tmpf $1
}

convert_model () {
	model_name=$1
	if [ -z "$2" ]
	then
		arg_mean_values=""
	else
		arg_mean_values="--mean_values ${2}"
	fi
	if [ -z "$3" ]
	then
		arg_scale_values=""
	else
		arg_scale_values="--scale_values ${3}"
	fi
	mean_values=$2
	scale_values=$3
	input_precision=$4
	tflite2tensorflow \
		--model_path ${model_name}.tflite \
		--model_output_path ${model_name} \
		--flatc_path ~/flatc \
		--schema_path ~/schema.fbs \
		--output_pb \
    	--optimizing_for_openvino_and_myriad
	# For generating Openvino "non normalized input" models (the normalization would need to be made explictly in the code):
	#tflite2tensorflow \
	#  --model_path ${model_name}.tflite \
	#  --model_output_path ${model_name} \
	#  --flatc_path ~/flatc \
	#  --schema_path ~/schema.fbs \
	#  --output_openvino_and_myriad
	# Generate Openvino "normalized input" models 
	/opt/intel/openvino_2021/deployment_tools/model_optimizer/mo_tf.py \
		--saved_model_dir ${model_name} \
		--model_name ${model_name} \
		--data_type FP16 \
		${arg_mean_values} \
		${arg_scale_values} \
		--reverse_input_channels
	# For Interpolate layers, replace in coordinate_transformation_mode, "half_pixel" by "align_corners"  (bug optimizer)
	if [ model_name != "pose_detection" ]
	then
		replace ${model_name}.xml half_pixel align_corners
	fi

	/opt/intel/openvino_2021/deployment_tools/inference_engine/lib/intel64/myriad_compile \
		-m ${model_name}.xml \
		-ip $input_precision \
		-VPU_NUMBER_OF_SHAVES 4 \
		-VPU_NUMBER_OF_CMX_SLICES 4 \
		-o ${model_name}_sh4.blob
}

convert_model pose_detection "[127.5,127.5,127.5]"  "[127.5,127.5,127.5]" "u8"
convert_model pose_landmark_full "" "" "fp16"
convert_model pose_landmark_lite "" "" "fp16"
convert_model pose_landmark_heavy "" "" "fp16"


