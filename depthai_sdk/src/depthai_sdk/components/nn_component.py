from .component import Component
from pathlib import Path
from typing import Optional, Union, List
import depthai as dai




class NNComponent(Component):

    _tracker: dai.node.ObjectTracker = None

    def __init__(self,
        pipeline: dai.Pipeline,
        model: Union[str, Path], # str for SDK supported model or Path to custom model's json
        input: Union[Component, dai.Node.Output],
        name: Optional[str] = None, # name of the node
        tracker: bool = False, # Enable object tracker - only for Object detection models
        spatial: bool = False, # 
        labels: Optional[List[int]] = None,
        out: bool = False,
        ) -> None:
        """
        Neural Network component that abstracts the following API nodes: NeuralNetwork, MobileNetDetectionNetwork,
        MobileNetSpatialDetectionNetwork, YoloDetectionNetwork, YoloSpatialDetectionNetwork, ObjectTracker
        (only for object detectors).

        Args:
            pipeline (dai.Pipeline)
            model (Union[str, Path]): str for SDK supported model or Path to custom model's json
            input: (Union[Component, dai.Node.Output]): Input to the NN. If nn_component that is object detector, crop HQ frame at detections (Script node + ImageManip node)
            name (Optional[str]): Name of the node
            tracker (bool, default False): Enable object tracker - only for Object detection models
            spatial (bool, default False): Enable getting Spatial coordinates (XYZ), only for for Obj detectors. Yolo/SSD use on-device spatial calc, others on-host (gen2-calc-spatials-on-host)
            labels (Optional[List[int]]): If input from NNComponent (object detector), crop & run inference only on objects with these labels
            out (bool, default False): Stream component's output to the host
        """
        # Create and link NN

        if tracker:
            self._tracker = pipeline.createObjectTracker()

        self.pipeline = pipeline

    def configTracker(self,
        type: Optional[dai.TrackerType] = None,
        trackLabels: Optional[List[int]] = None,
        assignmentPolicy: Optional[dai.TrackerIdAssignmentPolicy] = None,
        maxObj: Optional[int] = None,
        threshold: Optional[float] = None
        ):
        """
        Configure object tracker if it's enabled.

        Args:
            type (dai.TrackerType, optional): Set object tracker type
            trackLabels (List[int], optional): Set detection labels to track
            assignmentPolicy (dai.TrackerType, optional): Set object tracker ID assignment policy
            maxObj (int, optional): Set set max objects to track. Max 60.
            threshold (float, optional): Set threshold for object detection confidence. Default: 0.0
        """

        if self._tracker is None:
            raise Exception("Tracker wasn't enabled! Enable with cam.create_nn('[model]', tracker=True)")

        if type is not None:
            self._tracker.setTrackerType(type=type)
        if trackLabels is not None:
            self._tracker.setDetectionLabelsToTrack(trackLabels)
        if assignmentPolicy is not None:
            self._tracker.setTrackerIdAssignmentPolicy(assignmentPolicy)
        if maxObj is not None:
            if 60 < maxObj:
                raise ValueError("Maximum objects to track is 60!")
            self._tracker.setMaxObjectsToTrack(maxObj)
        if threshold is not None:
            self._tracker.setTrackerThreshold(threshold)
