import requests
import argparse
from argparse import ArgumentParser

def parse_args():
    epilog_text = '''
    Myriad blob compiler in cloud.
    -xml -bin -output
    '''
    parser = ArgumentParser(epilog=epilog_text,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-x", "--xml", default=None,
                        type=str, required=True,
                        help="XML from from Openvino IR")
    parser.add_argument("-b", "--bin", default=None,
                        type=str, required=True,
                        help="Bin from from Openvino IR")
    parser.add_argument("-o", "--output", default=None,
                        type=str, required=True,
                        help=".blob output")
    options = parser.parse_args()
    return options

args = vars(parse_args())


url = "http://69.164.214.171:8081"
payload = {'compile_flags': '-ip U8 -VPU_MYRIAD_PLATFORM VPU_MYRIAD_2480 -VPU_NUMBER_OF_SHAVES 4 -VPU_NUMBER_OF_CMX_SLICES 4'}
files = [
  ('definition', open(args['xml'],'rb')),
  ('weights', open(args["""bin"""],'rb'))
]
headers = {
  'Content-Type': 'application/json'
}
blob_file = open(args['output'],'wb')
response = requests.request("POST", url, data = payload, files = files)
blob_file.write(response.content)
blob_file.close()