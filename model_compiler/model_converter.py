import requests
import argparse
from argparse import ArgumentParser
from pathlib import Path
import subprocess


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
    parser.add_argument("-xml", "--xml", default=None,
                        type=str, required=True,
                        help="model name")
    parser.add_argument("-bin", "--bin", default=None,
                        type=str, required=True,
                        help="model name")
    options = parser.parse_args()
    return options

args = vars(parse_args())

def compile_model(xml, bin, shaves, cmx_slices, nces, output_file):
    PLATFORM="VPU_MYRIAD_2450" if nces == 0 else "VPU_MYRIAD_2480"

    # use 69.214.171 instead luxonis.com to bypass cloudflare limitation of max file size
    url = "http://69.164.214.171:8080/"
    payload = {
        'compile_type': 'myriad',
        'compiler_params': '-ip U8 -VPU_MYRIAD_PLATFORM ' + PLATFORM + ' -VPU_NUMBER_OF_SHAVES ' + str(shaves) +' -VPU_NUMBER_OF_CMX_SLICES ' + str(cmx_slices)
    }
    files = [
        ('definition', open(Path(xml), 'rb')),
        ('weights', open(Path(bin), 'rb'))
    ]
    try:
        response = requests.request("POST", url, data=payload, files=files)
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

def main():
    ret = compile_model(args['xml'], args['bin'], args['shaves'], args['cmx_slices'], args['NCEs'], args['output'])

    return ret

if __name__ == '__main__':
    ret = main()
    exit(ret)