Auto IR Brightness
==================

This example shows how to use the automatic IR brightness feature of the DepthAI Stereo Camera. 
The function ``set_auto_ir(auto_mode=True)`` enables/disables auto IR dot projector and flood brightness. If enabled, it selects the best IR brightness level automatically.

Can be set to continious mode, which will continuously adjust the IR brightness. Set to ``False`` by default and which will automatically adjust the IR brightness only at device bootup.

.. include::  /includes/blocking_behavior.rst
    

Setup
#####

.. include::  /includes/install_from_pypi.rst

Pipeline
########

.. image:: /_static/images/pipelines/stereo_auto_ir.png
      :alt: Pipeline graph



Source Code
###########

.. tabs::

    .. tab:: Python

        Also `available on GitHub <https://github.com/luxonis/depthai/tree/main/depthai_sdk/examples/StereoComponent/stereo_auto_ir.py>`_.

        .. literalinclude:: ../../../../examples/StereoComponent/stereo_auto_ir.py
            :language: python
            :linenos:

.. include::  /includes/footer-short.rst