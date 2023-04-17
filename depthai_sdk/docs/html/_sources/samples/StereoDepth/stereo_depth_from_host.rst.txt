Stereo Depth from host
======================

This example shows depth map from host using stereo images. There are 3 depth modes which you can select inside the code:

#. `lr_check`: used for better occlusion handling. For more information `click here <https://docs.luxonis.com/en/latest/pages/faq/#left-right-check-depth-mode>`__
#. `extended_disparity`: suitable for short range objects. For more information `click here <https://docs.luxonis.com/en/latest/pages/faq/#extended-disparity-depth-mode>`__
#. `subpixel`: suitable for long range. For more information `click here <https://docs.luxonis.com/en/latest/pages/faq/#subpixel-disparity-depth-mode>`__

Otherwise a median with kernel_7x7 is activated.

.. rubric:: Similar samples:

- :ref:`Stereo Depth Video`

Setup
#####

.. include::  /includes/install_from_pypi.rst

This example also requires dataset folder - you can download it from
`here <https://drive.google.com/file/d/1mPyghc_0odGtSU2cROS1uLD-qwRU8aug>`__

Source code
###########

.. tabs::

    .. tab:: Python

        Also `available on GitHub <https://github.com/luxonis/depthai-python/blob/main/examples/StereoDepth/stereo_depth_from_host.py>`__

        .. literalinclude:: ../../../../examples/StereoDepth/stereo_depth_from_host.py
           :language: python
           :linenos:

.. include::  /includes/footer-short.rst
