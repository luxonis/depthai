# This script has to be run from the docker container started by ./docker_tflite2tensorflow.sh

usage ()
{
	echo "Generate a new blob with a specified number of shaves and cmx (nb cmx = nb shaves)"
	echo "Usage: ${0} -m model -n nb_shaves"
	echo "model = 'pd' for pose detection, 'full' or 'lite' or 'heavy' for landmarks"
	echo "nb_shaves must be between 1 and 13"
}

while getopts ":hm:n:" opt; do
	case ${opt} in
		h )
			usage
			exit 0
			;;
		m )
			model=$OPTARG
			;;
		n )
			nb_shaves=$OPTARG
			;;
		: )
			echo "Error: -$OPTARG requires an argument."
			usage
			exit 1
			;;
		\? )
			echo "Invalid option: -$OPTARG" 
			usage
			exit 1
			;;
	esac
done
if [ -z "$model" ]
then
	echo "Model not specified !"
	usage
	exit 1
fi
if [ $model == "pd" ]
then
	model="pose_detection"
	input_precision="u8"
elif [ $model == "full" ]
then
	model="pose_landmark_full"
	input_precision="fp16"
elif [ $model == "lite" ]
then
	model="pose_landmark_lite"
	input_precision="fp16"
elif [ $model == "heavy" ]
then
	model="pose_landmark_heavy"
	input_precision="fp16"
else
	echo "Invalid model !"
	usage
	exit 1
fi
model_xml="${model}.xml"
if [ ! -f $model_xml ]
then
	echo "The model ${model_xml} does not exist"
	echo "Have you run convert_models.sh ?"
	exit 1
fi
if [ -z "$nb_shaves" ]
then
	echo "The number of shaves must be specified with -n"
	usage
	exit 1
fi
if [ $nb_shaves -lt 1 -o $nb_shaves -gt 13 ]
then
	echo "Invalid number of shaves !"
	usage
	exit 1
fi

model_blob="${model}_sh${nb_shaves}.blob"
	
echo Model: $model_xml $model_blob
echo Shaves $nb_shaves

source /opt/intel/openvino_2021/bin/setupvars.sh


/opt/intel/openvino_2021/deployment_tools/inference_engine/lib/intel64/myriad_compile -m $model_xml -ip $input_precision -VPU_NUMBER_OF_SHAVES $nb_shaves -VPU_NUMBER_OF_CMX_SLICES $nb_shaves -o $model_blob

