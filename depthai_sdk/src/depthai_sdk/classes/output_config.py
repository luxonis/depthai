from typing import Optional, Callable, Union, Tuple


class VisualizeConfig:
    # TODO: support visualziation configs. eg. colors, fonts, locations where text in BBs is displayed,
    # BB rectangle config (transparency, rounded edges etc.)
    scale: Union[None, float, Tuple[int, int]]
    fps: bool
    def __init__(self, scale, fps):
        self.scale = scale
        self.fps = fps

class OutputConfig:
    """
    Saves callbacks/visualizers until the device is fully initialized. I'll admit it's not the cleanest solution.
    """
    vis: Optional[VisualizeConfig] # Visualization
    output: Callable # Output of the component (a callback)
    callback: Callable # Callback that gets called after syncing

    def __init__(self, output: Callable, callback: Callable, vis: VisualizeConfig = None):
        self.output = output
        self.callback = callback
        self.vis = vis
