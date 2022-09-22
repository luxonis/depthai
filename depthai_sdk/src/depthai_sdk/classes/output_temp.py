from typing import Optional, Callable

from ..visualizing import VisualizerManager


# TODO: rename
class OutputTemplate:
    """
    Saves callbacks/visualizers until the device is fully initialized. I'll admit it's not the cleanest solution.
    """
    vis: Optional[VisualizerManager] # Visualization
    output: Callable # Output of the component (a callback)
    callback: Callable # Callback that gets called after syncing

    def __init__(self, output: Callable, callback: Callable, vis: VisualizerManager = None):
        self.output = output
        self.callback = callback
        self.vis = vis
