import subprocess

def download_model(model, shaves, cmx_slices, nces, output_file):
    ret = subprocess.call(['model_compiler/download_and_compile_cloud.sh', str(model), str(shaves), str(cmx_slices), str(nces)])
    if(ret != 0):
        print("Model compilation error!")

    return ret