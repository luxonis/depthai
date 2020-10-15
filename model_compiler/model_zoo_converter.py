import requests
import argparse
from argparse import ArgumentParser


def parse_args():
    epilog_text = '''
    Myriad blob compiler in cloud.
    '''
    parser = ArgumentParser(epilog=epilog_text,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-sh", "--shaves", default=4, type=int,
                        help="Number of shaves used by NN.")
    parser.add_argument("-cmx", "--cmx_slices", default=4, type=int,
                        help="Number of cmx slices used by NN.")
    parser.add_argument("-NCE", "--NCEs", default=1, type=int,
                        help="Number of NCEs used by NN.")
    parser.add_argument("-o", "--output", default=None,
                        type=str, required=True,
                        help=".blob output")
    parser.add_argument("-i", "--input", default=None,
                        type=str, required=True,
                        help="model name")
    options = parser.parse_args()
    return options

args = vars(parse_args())

def download_model(model, shaves, cmx_slices, nces, output_file):
    PLATFORM="VPU_MYRIAD_2450" if nces == 0 else "VPU_MYRIAD_2480"

    url = "http://luxonis.com:8080/"
    payload = {
        'compile_type': 'zoo',
        'model_name': model,
        'model_downloader_params': '--precisions FP16 --num_attempts 5',
        'intermediate_compiler_params': '--data_type=FP16 --mean_values [127.5,127.5,127.5] --scale_values [255,255,255]',
        'compiler_params': '-ip U8 -VPU_MYRIAD_PLATFORM ' + PLATFORM + ' -VPU_NUMBER_OF_SHAVES ' + str(shaves) +' -VPU_NUMBER_OF_CMX_SLICES ' + str(cmx_slices)
    }
    try:
        response = requests.request("POST", url, data=payload)
    except:
        print("Connection timed out!")
        return 1
    if response.status_code == 200:
        blob_file = open(output_file,'wb')
        blob_file.write(response.content)
        blob_file.close()
    else:
        print("Model compilation failed with error code: " + str(response.status_code))
        print(str(response.text.encode('utf8')))
        return 2

    return 0

ret = download_model(args['input'], args['shaves'], args['cmx_slices'], args['NCEs'], args['output'])

exit(ret)