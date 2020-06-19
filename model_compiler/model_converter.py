import requests
import argparse
from argparse import ArgumentParser


def parse_args():
    epilog_text = '''
    Myriad blob compiler in cloud.
    '''
    parser = ArgumentParser(epilog=epilog_text,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-sh", "--shaves", default=4, type=str,
                        help="Number of shaves used by NN.")
    parser.add_argument("-cmx", "--cmx_slices", default=4, type=str,
                        help="Number of cmx slices used by NN.")
    parser.add_argument("-o", "--output", default=None,
                        type=str, required=True,
                        help=".blob output")
    parser.add_argument("-i", "--input", default=None,
                        type=str, required=True,
                        help="model name")
    options = parser.parse_args()
    return options

args = vars(parse_args())


url = "http://69.164.214.171:8080/"
payload = {
    'compile_type': 'zoo',
    'model_name': args['input'],
    'model_downloader_params': '--precisions FP16 --num_attempts 5',
    'intermediate_compiler_params': '--data_type=FP16 --mean_values [127.5,127.5,127.5] --scale_values [255,255,255]',
    'compiler_params': '-ip U8 -VPU_MYRIAD_PLATFORM VPU_MYRIAD_2480 -VPU_NUMBER_OF_SHAVES ' + args['shaves'] +' -VPU_NUMBER_OF_CMX_SLICES ' + args['cmx_slices']
}
headers = {
    'Content-Type': 'application/json'
}

try:
  blob_file = open(args['output'],'wb')
  response = requests.request("POST", url,  data=payload)
  # print(response.text.encode('utf8'))
  blob_file.write(response.content)
  blob_file.close()
except:
  print("Connection timed out!")
  exit(1)
