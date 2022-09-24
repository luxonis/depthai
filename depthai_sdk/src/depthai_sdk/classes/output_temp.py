from typing import Optional, Callable, Union, Tuple


class VisualizeTemplate:
    scale: Union[None, float, Tuple[int, int]]
    fps: bool
    def __init__(self, scale, fps):
        self.scale = scale
        self.fps = fps

# TODO: rename
class OutputTemplate:
    """
    Saves callbacks/visualizers until the device is fully initialized. I'll admit it's not the cleanest solution.
    """
    vis: Optional[VisualizeTemplate] # Visualization
    output: Callable # Output of the component (a callback)
    callback: Callable # Callback that gets called after syncing

    def __init__(self, output: Callable, callback: Callable, vis: VisualizeTemplate = None):
        self.output = output
        self.callback = callback
        self.vis = vis
