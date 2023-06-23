from depthai_sdk import OakCamera

# Download & deploy a model from Roboflow universe:
# # https://universe.roboflow.com/david-lee-d0rhs/american-sign-language-letters/dataset/6

with OakCamera() as oak:
    color = oak.create_camera('color')
    model_config = {
        'source': 'roboflow', # Specify that we are downloading the model from Roboflow
        'model':'american-sign-language-letters/6',
        'key':'181b0f6e43d59ee5ea421cd77f6d9ea2a4b059f8' # Fake API key, replace with your own!
    }
    nn = oak.create_nn(model_config, color)
    oak.visualize(nn, fps=True)
    oak.start(blocking=True)
