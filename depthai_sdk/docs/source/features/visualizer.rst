Visualizer
==========

Visualizer helps user to visualize the output of AI models in a more intuitive and customizable way.

Getting Started
#################

``Visualizer`` is created upon calling ``OakCamera.visualize(...)``, which returns ``Visualizer`` instance.
Once it is created, the visualizer configs can be modified using ``Visualizer.configure_{output,text,bbox,tracking}`` methods.

Example how ``Visualizer`` can be created:

.. code-block:: python

    with OakCamera() as oak:
        cam = oak.create_camera(...)
        visualizer = cam.visualize(cam.out.main)

Visualizer automatically detects the platform the application is running on and chooses the way to handle the output.
At the moment, it distinguishes between 2 platforms:

* PC, where the output is displayed in a window.
* RobotHub, where the output is shared with the RobotHub application.

The visualizer is primarily used in :ref:`Packets` in ``visualize`` methods.

Configs
#################

``Visualizer`` contains 4 types of configs: ``output``, ``text``, ``detections``, and ``tracking``.
Each config's type has its own set of parameters, which effects how the corresponding object will be visualized.

There are the following methods for modifying the default configuration: ``Visualizer.configure_{output,text,detections,tracking}``.
The arguments should be passed as key-value arguments with the same signature as the corresponding config,
e.g., ``configure_text(font_size=2, font_color=(255,123,200))``.

The modified configuration will be applied to every created objects. The ``configure_*()`` methods support
fluent interface and can be chained, e.g., ``configure_text(font_size=2).configure_bbox(color=(255, 0, 0))``.

Example how to configure the visualizer:

.. code-block:: python

        visualizer = oak.visualize(camera.out.main)
        visualizer.configure_bbox(
            bbox_style=BboxStyle.RECTANGLE,
            label_position=TextPosition.MID,
        ).configure_text(
            auto_scale=True
        )

.. automodule:: depthai_sdk.visualize.configs
    :members:
    :undoc-members:

Objects
#################

Visualizer operates with objects. All objects must be derived from ``GenericObject`` class.
Objects can be added to the visualizer using ``Visualizer.add_{text,detection,trail}(...)`` method.

Visualizer supports the following objects:

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

Create your own object
##############

If the provided functionality is not enough, you can create your own object. To do so, you need to create a class
derived from ``GenericObject`` and implement the ``prepare()``, ``serialize()``, and ``draw()`` methods.
The ``draw()`` method should draw the object on the passed ``frame`` argument.

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
##############

The following script will visualize the output of face detection model.

.. code-block:: python

    from depthai_sdk import OakCamera
    from depthai_sdk.visualize.configs import BboxStyle, TextPosition

    with OakCamera() as oak:
        camera = oak.create_camera('color')

        det = oak.create_nn('face-detection-retail-0004', camera)

        visualizer = oak.visualize(det.out.main)
        visualizer.configure_bbox(
            color=(0, 255, 0),
            thickness=2,
            bbox_style=BboxStyle.RECTANGLE,
            label_position=TextPosition.MID,
        ).configure_text(
            font_color=(255, 255, 0),
            auto_scale=True
        ).configure_output(
            show_fps=True,
        ).configure_tracking(
            line_thickness=5
        )

        oak.start(blocking=True)

References
##############

...