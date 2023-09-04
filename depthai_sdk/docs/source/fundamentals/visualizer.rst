Visualizer
==========

DepthAI SDK visualizer serves as a tool to visualize the output of the DepthAI pipeline.

It can be used to visualize the output of the camera, neural network, depth and disparity map, the rectified streams, the spatial location of the detected objects, and more.

Getting Started
###############

:class:`Visualizer <depthai_sdk.visualize.visualizer.Visualizer>` is created upon calling
:meth:`OakCamera.visualize() <depthai_sdk.OakCamera.visualize>`, which returns :func:`Visualizer <depthai_sdk.visualize.visualizer.Visualizer>` instance.
Once it is created, the visualizer configs can be modified using :meth:`output() <depthai_sdk.visualize.visualizer.Visualizer.output>`,
:meth:`stereo() <depthai_sdk.visualize.visualizer.Visualizer.stereo>`,
:meth:`text() <depthai_sdk.visualize.visualizer.Visualizer.text>`,
:meth:`detections() <depthai_sdk.visualize.visualizer.Visualizer.detections>`,
:meth:`tracking() <depthai_sdk.visualize.visualizer.Visualizer.tracking>` methods.

Example how :class:`Visualizer <depthai_sdk.visualize.visualizer.Visualizer>` can be created:

.. code-block:: python

    from depthai_sdk import OakCamera

    with OakCamera() as oak:
        cam = oak.create_camera('color')
        visualizer = oak.visualize(cam.out.main)
        oak.start(blocking=True)

:func:`Visualizer <depthai_sdk.visualize.visualizer.Visualizer>` is primarily used alongside with :ref:`Packets`
in :mod:`depthai_sdk.oak_outputs` module.

Configs
#######

:func:`Visualizer <depthai_sdk.visualize.visualizer.Visualizer>` is configurable via
:class:`VisConfig <depthai_sdk.visualize.configs.VisConfig>` that consists of five auxiliary configs:
:class:`OutputConfig <depthai_sdk.visualize.configs.OutputConfig>`,
:class:`StereoConfig <depthai_sdk.visualize.configs.StereoConfig>`,
:class:`TextConfig <depthai_sdk.visualize.configs.TextConfig>`,
:class:`DetectionConfig <depthai_sdk.visualize.configs.DetectionConfig>`,
and :class:`TrackingConfig <depthai_sdk.visualize.configs.TrackingConfig>`.
Each config's type has its own set of parameters, which effects how the corresponding object will be visualized.

There are the following methods for modifying the default configuration:
:meth:`output() <depthai_sdk.visualize.visualizer.Visualizer.output>`,
:meth:`stereo() <depthai_sdk.visualize.visualizer.Visualizer.stereo>`,
:meth:`text() <depthai_sdk.visualize.visualizer.Visualizer.text>`,
:meth:`detections() <depthai_sdk.visualize.visualizer.Visualizer.detections>`,
:meth:`tracking() <depthai_sdk.visualize.visualizer.Visualizer.tracking>`.
The arguments should be passed as key-value arguments with the same signature as the corresponding config,
e.g., :meth:`Visualizer.text(font_size=2, font_color=(255,123,200)) <depthai_sdk.visualize.visualizer.Visualizer.text>`.

The modified configuration will be applied to every created objects. The methods support
fluent interface and can be chained, e.g., :meth:`Visualizer.text(font_size=2).detections(color=(255, 0, 0)) <depthai_sdk.visualize.visualizer.Visualizer.text>`.

Example how to configure the visualizer:

.. code-block:: python

        visualizer = oak.visualize(camera.out.main)
        visualizer.detections(
            bbox_style=BboxStyle.RECTANGLE,
            label_position=TextPosition.MID,
        ).text(
            auto_scale=True
        )

Objects
#######

:class:`Visualizer <depthai_sdk.visualize.visualizer.Visualizer>` operates with objects. Objects can be seen as a hierarchical structure.
The root object is ``self``, and the children are the list of the created objects.
:meth:`add_child <depthai_sdk.visualize.objects.GenericObject.add_child>` should be used to add the object to the children list.
The parent object shares the config and frame shape with all children.

All objects must be derived from :class:`GenericObject <depthai_sdk.visualize.objects.GenericObject>`.

Implemented objects:

* :class:`VisDetections <depthai_sdk.visualize.objects.VisDetections>`,
* :class:`VisText <depthai_sdk.visualize.objects.VisText>`,
* :class:`VisLine <depthai_sdk.visualize.objects.VisLine>`,
* :class:`VisCircle <depthai_sdk.visualize.objects.VisCircle>`,
* :class:`VisTrail <depthai_sdk.visualize.objects.VisTrail>`.

Objects can be added to the visualizer using the following methods:

* :meth:`add_text() <depthai_sdk.visualize.visualizer.Visualizer.add_text>`,
* :meth:`add_detections() <depthai_sdk.visualize.visualizer.Visualizer.add_detections>`,
* :meth:`add_trail() <depthai_sdk.visualize.visualizer.Visualizer.add_trail>`,
* :meth:`add_circle() <depthai_sdk.visualize.visualizer.Visualizer.add_circle>`,
* :meth:`add_line() <depthai_sdk.visualize.visualizer.Visualizer.add_line>`.

Create your own object
######################

If the provided functionality is not enough, you can create your own object. To do so, you need to create a class
derived from :class:`GenericObject <depthai_sdk.visualize.objects.GenericObject>` and implement the
:meth:`prepare <depthai_sdk.visualize.objects.GenericObject.prepare>`,
:meth:`serialize <depthai_sdk.visualize.objects.GenericObject.serialize>`,
and :meth:`draw <depthai_sdk.visualize.objects.GenericObject.draw>` methods.
The :meth:`draw <depthai_sdk.visualize.objects.GenericObject.draw>` method should draw the object on the passed ``frame`` argument.

.. code-block:: python

    class YourOwnObject:
        def __init__(self, ...):
            ...

        def prepare(self) -> None:
            ...

        def serialize(self) -> str:
            ...

        def draw(self, frame) -> None:
            ...

    with OakCamera() as oak:
        cam = oak.create_camera(...)
        visualizer = cam.visualize(cam.out.main)
        visualizer.add_object(YourOwnObject(...))

Example usage
#############

The following script will visualize the output of face detection model:

.. code-block:: python

    from depthai_sdk import OakCamera
    from depthai_sdk.visualize.configs import BboxStyle, TextPosition

    with OakCamera() as oak:
        camera = oak.create_camera('color')

        det = oak.create_nn('face-detection-retail-0004', camera)

        visualizer = oak.visualize(det.out.main, fps=True)
        visualizer.detections(
            color=(0, 255, 0),
            thickness=2,
            bbox_style=BboxStyle.RECTANGLE,
            label_position=TextPosition.MID,
        ).text(
            font_color=(255, 255, 0),
            auto_scale=True
        ).tracking(
            line_thickness=5
        )

        oak.start(blocking=True)

Serialization
#############
The visualizer provides a way to serialize the output objects to JSON, which can be used for further processing.

JSON schemas
++++++++++++

General config
~~~~~~~~~~~~~~

.. literalinclude:: ../visualizer_formats/format.json
   :language: json

Objects
~~~~~~~

- Detection:

.. literalinclude:: ../visualizer_formats/detection_format.json
   :language: json

- Text:

.. literalinclude:: ../visualizer_formats/text_format.json
   :language: json

- Line:

.. literalinclude:: ../visualizer_formats/line_format.json
   :language: json

Example JSON output
~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../visualizer_formats/example.json
   :language: json


References
##########

Visualizer
----------

.. autoclass:: depthai_sdk.visualize.visualizer.Visualizer
    :members:
    :undoc-members:

Objects
-------

.. autoclass:: depthai_sdk.visualize.objects.GenericObject
    :members:
    :undoc-members:

.. autoclass:: depthai_sdk.visualize.objects.VisDetections
    :members:
    :undoc-members:

.. autoclass:: depthai_sdk.visualize.objects.VisText
    :members:
    :undoc-members:

.. autoclass:: depthai_sdk.visualize.objects.VisLine
    :members:
    :undoc-members:

.. autoclass:: depthai_sdk.visualize.objects.VisTrail
    :members:
    :undoc-members:

Configs
-------

.. automodule:: depthai_sdk.visualize.configs
    :members:
    :undoc-members:

.. include::  ../includes/footer-short.rst
