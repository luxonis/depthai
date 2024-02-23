OakCamera
=========

The **OakCamera** class abstracts:

- DepthAI API pipeline building with :ref:`Components`.
- Stream :ref:`recording <Recording>` and :ref:`replaying <Replaying>`.
- Debugging features (such as ``oak.show_graph()``).
- :ref:`AI model <AI models>` sourcing and decoding.
- Message syncing & visualization, and much more.

.. note::
    This class will be in **alpha stage** until **depthai-sdk 2.0.0**, so there will likely be some API changes.


Interoperability with DepthAI API
---------------------------------

DepthAI SDK was developed with `DepthAI API <https://docs.luxonis.com/projects/api/en/latest/>`__ interoperability in mind.
Users can access all depthai API nodes inside components, along with the `dai.Pipeline <https://docs.luxonis.com/projects/api/en/latest/components/pipeline/>`__ (``oak.pipeline``)
and `dai.Device <https://docs.luxonis.com/projects/api/en/latest/components/device/>`__ (``oak.device``) objects.

.. literalinclude:: ../../examples/mixed/api_interop.py
   :language: python

Examples
--------

Below there are a few basic examples. **See** `all examples here <https://github.com/luxonis/depthai/tree/main/depthai_sdk/examples>`__.

Here are a few demos that have been developed with DepthAI SDK:

#. `age-gender <https://github.com/luxonis/depthai-experiments/tree/master/gen2-age-gender>`__,
#. `emotion-recognition <https://github.com/luxonis/depthai-experiments/tree/master/gen2-emotsion-recognition>`__,
#. `full-fov-nn <https://github.com/luxonis/depthai-experiments/tree/master/gen2-full-fov-nn>`__,
#. `head-posture-detection <https://github.com/luxonis/depthai-experiments/tree/master/gen2-head-posture-detection>`__,
#. `pedestrian-reidentification <https://github.com/luxonis/depthai-experiments/tree/master/gen2-pedestrian-reidentification>`__,
#. `people-counter <https://github.com/luxonis/depthai-experiments/tree/master/gen2-people-counter>`__,
#. `people-tracker <https://github.com/luxonis/depthai-experiments/tree/master/gen2-people-tracker>`__,
#. `mask-detection <https://github.com/luxonis/depthai-experiments/tree/master/gen2-mask-detection>`__,
#. `yolo <https://github.com/luxonis/depthai-experiments/tree/master/gen2-yolo>`__.
#. `Roboflow <https://blog.roboflow.com/deploy-roboflow-model-luxonis-depth-sdk>`__.


Preview color and mono cameras
******************************

.. literalinclude:: ../../examples/CameraComponent/rgb_mono_preview.py
   :language: python


Run MobilenetSSD on color camera
********************************

.. literalinclude:: ../../examples/NNComponent/mobilenet.py
   :language: python

Run face-detection-retail-0004 on left camera
*********************************************

.. literalinclude:: ../../examples/NNComponent/face_detection_left.py
   :language: python

Deploy models from Roboflow and Roboflow Universe with Depth SDK
****************************************************************

.. literalinclude:: ../../examples/NNComponent/roboflow_integration.py
   :language: python


Reference
#########

.. autoclass:: depthai_sdk.OakCamera
    :members:
    :undoc-members:

.. include::  includes/footer-short.rst
