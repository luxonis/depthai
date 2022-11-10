from depthai_sdk import OakCamera
API_KEY = 'd2OP8nbhA9rZcWd6G8p1' # Fake API key, replace with your own!

with OakCamera() as oak:
    color = oak.create_camera('color')
    model_config = {
        'source': 'roboflow', # We might later support other ML training platforms
        'model':'american-sign-language-letters/6',
        'key':API_KEY
    }
    nn = oak.create_nn(model_config, color)
    oak.visualize(nn, fps=True)
    oak.start(blocking=True)