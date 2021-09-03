Getting started with DepthAI SDK
================================

In this tutorial, we'll show you how to use DepthAI SDK for a couple of basic use cases, that can give you an overall
idea how to use it and in which cases it might be useful.

What is DepthAI SDK?
--------------------

DepthAI SDK was created on top of the regular DepthAI API. Originally, it was a part of the `demo script <https://github.com/luxonis/depthai/blob/main/depthai_demo.py>`__,
but over time it evolved to become a package containing many convenience methods and classes that aim to help in development process with OAK cameras.

Package is mainly made of **managers**, which handle different aspects of development:

.. list-table::
  :widths: 50 50

  * - :class:`depthai.managers.PipelineManager`
    - Helps in setting up processing pipeline
  * - :class:`depthai.managers.NNetManager`
    - Helps in setting up neural networks
  * - :class:`depthai.managers.PreviewManager`
    - Helps in displaying preview from OAK cameras
  * - :class:`depthai.managers.EncodingManager`
    - Helps in creating videos from OAK cameras
  * - :class:`depthai.managers.BlobManager`
    - Helps in downloading neural networks as MyriadX blobs

Together with managers, you can use:

.. list-table::
  :widths: 50 50

  * - :class:`depthai.fps`
    - For FPS calculations
  * - :class:`depthai.previews`
    - For frame handling
  * - :class:`depthai.utils`
    - For various most-common tasks

In some places, code is also adjusted for modifications - e.g. you can set up a custom handler file for neural network
or pass a callback argument to a function to perform additional modifications

Example usages
--------------

The original "user" of this SDK was the `demo script <https://github.com/luxonis/depthai/blob/main/depthai_demo.py>`__, where you can see how the SDK is used.
Below, you can find a list of other projects that also use the SDK and are available to use as a reference

* `<https://github.com/luxonis/depthai-experiments/tree/sdk/gen2-human-pose>`__

Installation
------------

.. include::  install.rst

Cookbook
--------

Below you can find various basic usages of DepthAI SDK that can be used as a starting point. For more in-depth informations
about the classes, please visit :ref:`DepthAI SDK API`

Preview color camera
********************

.. code-block:: python

    from depthai_sdk import Previews
    from depthai_sdk.managers import PipelineManager, PreviewManager
    import depthai as dai
    import cv2

    pm = PipelineManager()
    pm.create_color_cam(xout=True)

    with dai.Device(pm.pipeline) as device:
        pv = PreviewManager(display=[Previews.color.name])
        pv.create_queues(device)

        while True:
            pv.prepare_frames()
            pv.show_frames()

            if cv2.waitKey(1) == ord('q'):
                break

Preview color and mono cameras
******************************

.. code-block:: python

    from depthai_sdk import Previews
    from depthai_sdk.managers import PipelineManager, PreviewManager
    import depthai as dai
    import cv2

    pm = PipelineManager()
    pm.create_color_cam(xout=True)
    pm.create_left_cam(xout=True)
    pm.create_right_cam(xout=True)

    with dai.Device(pm.pipeline) as device:
        pv = PreviewManager(display=[Previews.color.name, Previews.left.name, Previews.right.name])
        pv.create_queues(device)

        while True:
            pv.prepare_frames()
            pv.show_frames()

            if cv2.waitKey(1) == ord('q'):
                break

Run MobilenetSSD on color camera
********************************

.. code-block:: python

    from depthai_sdk import Previews
    from depthai_sdk.managers import PipelineManager, PreviewManager, NNetManager, BlobManager
    import depthai as dai
    import cv2

    pm = PipelineManager()
    pm.create_color_cam(preview_size=(300, 300), xout=True)

    bm = BlobManager(zoo_name="mobilenet-ssd")
    nm = NNetManager(input_size=(300, 300), nn_family="mobilenet")
    nn_pipeline = nm.create_nn_pipeline(pipeline=pm.pipeline, nodes=pm.nodes, source=Previews.color.name,
        blob_path=bm.getBlob(shaves=6, openvino_version=pm.pipeline.getOpenVINOVersion())
    )
    pm.add_nn(nn_pipeline)

    with dai.Device(pm.pipeline) as device:
        pv = PreviewManager(display=[Previews.color.name])
        pv.create_queues(device)
        nm.createQueues(device)
        nn_data = []

        while True:
            pv.prepare_frames()
            in_nn = nm.output_queue.tryGet()

            if in_nn is not None:
                nn_data = nm.decode(in_nn)

            nm.draw(pv, nn_data)
            pv.show_frames()

            if cv2.waitKey(1) == ord('q'):
                break

Run face-detection-retail-0004 on left camera
*********************************************

.. code-block:: python

    from depthai_sdk import Previews
    from depthai_sdk.managers import PipelineManager, PreviewManager, NNetManager, BlobManager
    import depthai as dai
    import cv2

    pm = PipelineManager()
    pm.create_left_cam(xout=True)

    bm = BlobManager(zoo_name="face-detection-retail-0004")
    nm = NNetManager(input_size=(300, 300), nn_family="mobilenet")
    nn_pipeline = nm.create_nn_pipeline(pipeline=pm.pipeline, nodes=pm.nodes, source=Previews.left.name,
        blob_path=bm.getBlob(shaves=6, openvino_version=pm.pipeline.getOpenVINOVersion())
    )
    pm.add_nn(nn_pipeline)

    with dai.Device(pm.pipeline) as device:
        pv = PreviewManager(display=[Previews.left.name])
        pv.create_queues(device)
        nm.createQueues(device)
        nn_data = []

        while True:
            pv.prepare_frames()
            in_nn = nm.output_queue.tryGet()

            if in_nn is not None:
                nn_data = nm.decode(in_nn)

            nm.draw(pv, nn_data)
            pv.show_frames()

            if cv2.waitKey(1) == ord('q'):
                break


.. include::  footer-short.rst




