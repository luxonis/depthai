from depthai_sdk.oak_outputs.xout.xout_base import StreamXout, XoutBase
from depthai_sdk.oak_outputs.xout.xout_location import XoutLocation
from depthai_sdk.components.stereo_component import StereoComponent
from depthai_sdk.components.component import Component

from typing import Union
import depthai as dai

class SpatialLocationCalculatorComponent(Component):
    def __init__(self, pipeline: dai.Pipeline, stereo: Union[StereoComponent, None]):
        self.out = self.Out(self)

        if not isinstance(stereo, StereoComponent):
            return
        self.node = pipeline.createSpatialLocationCalculator()
        self._spatialLocation = (0, 0, 0)
        self._configQueue = None
        self._stereo = stereo

        if not isinstance(self._stereo, StereoComponent):
            return
        topLeft = dai.Point2f(0.4, 0.4)
        bottomRight = dai.Point2f(0.6, 0.6)

        config = dai.SpatialLocationCalculatorConfigData()
        config.depthThresholds.lowerThreshold = 100
        config.depthThresholds.upperThreshold = 10000
        config.roi = dai.Rect(topLeft, bottomRight)

        self.node.inputConfig.setWaitForMessage(False)
        self.node.initialConfig.addROI(config)
        self._stereo.node.depth.link(self.node.inputDepth)

        self._xin = pipeline.createXLinkIn()
        self._xin.setStreamName("spatial_location_calculator_config")
        self._xin.out.link(self.node.inputConfig)

    def _store_locations(self, spatialLocations: dai.SpatialLocationCalculatorData):
        for location in spatialLocations.getSpatialLocations(): # Doesn't handle more than one right now
            self._spatialLocation = location

    def _get_location(self):
        if isinstance(self._spatialLocation, dai.SpatialLocations):
            coords = self._spatialLocation.spatialCoordinates
            return (coords.x, coords.y, coords.z)
        return self._spatialLocation

    def configure(self, topLeft=(0, 0), botRight=(0, 0), lowerThreshold=100, upperThreshold=10000):
        if self._configQueue == None:
            return
        config = dai.SpatialLocationCalculatorConfigData()
        config.depthThresholds.lowerThreshold = lowerThreshold
        config.depthThresholds.upperThreshold = upperThreshold
        config.roi = dai.Rect(topLeft[0], topLeft[1], botRight[0], botRight[1])
        config.calculationAlgorithm = dai.SpatialLocationCalculatorAlgorithm.AVERAGE
        cfg = dai.SpatialLocationCalculatorConfig()
        cfg.addROI(config)
        try:
            self._configQueue.send(cfg)
        except:
            pass

    def get_stream_xout(self) -> StreamXout:
        return StreamXout(id=self.node.id, out=self.node.out, name="SpatialLocationCalculator")

    class Out:
        def __init__(self, spatialLocationCalculatorComponent: 'SpatialLocationCalculatorComponent'):
            self._comp = spatialLocationCalculatorComponent

        def main(self, pipeline: dai.Pipeline, device: dai.Device) -> XoutBase:
            return self._comp._create_xout(pipeline, XoutLocation(device, self._comp, self._comp.get_stream_xout()))