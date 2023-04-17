Frame syncing on OAK
====================

This example showcases how you can use :ref:`Script` node to perform :ref:`Message syncing` of multiple streams.
Example uses :ref:`ImgFrame`'s timestamps to achieve syncing precision.

Similar syncing demo scripts (python) can be found at our depthai-experiments repository in `gen2-syncing <https://github.com/luxonis/depthai-experiments/tree/master/gen2-syncing>`__
folder.

Demo
####

Terminal log after about 13 minutes. Color and disparity streams are perfectly in-sync.

.. code-block:: bash

[1662574807.8811488] Stream rgb, timestamp: 7:26:21.601595, sequence number: 21852
[1662574807.8821492] Stream disp, timestamp: 7:26:21.601401, sequence number: 21852

[1662574807.913144] Stream rgb, timestamp: 7:26:21.634982, sequence number: 21853
[1662574807.9141443] Stream disp, timestamp: 7:26:21.634730, sequence number: 21853

[1662574807.9451444] Stream rgb, timestamp: 7:26:21.668243, sequence number: 21854
[1662574807.946151] Stream disp, timestamp: 7:26:21.668057, sequence number: 21854

Setup
#####

.. include::  /includes/install_from_pypi.rst

.. include:: /includes/install_req.rst

Source code
###########

.. tabs::

    .. tab:: Python

        Also `available on GitHub <https://github.com/luxonis/depthai-python/blob/main/examples/mixed/frame_sync.py>`__

        .. literalinclude:: ../../../../examples/mixed/frame_sync.py
           :language: python
           :linenos:

    .. tab:: C++

        Also `available on GitHub <https://github.com/luxonis/depthai-core/blob/main/examples/mixed/frame_sync.cpp>`__

        .. literalinclude:: ../../../../depthai-core/examples/mixed/frame_sync.cpp
           :language: cpp
           :linenos:

.. include::  /includes/footer-short.rst
