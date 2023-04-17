Script HTTP server
==================

.. note::
    This example can only run on `OAK POE devices <https://docs.luxonis.com/projects/hardware/en/latest/#poe-designs>`__. You need **bootloader** on/above version **0.0.15**. You can flash bootloader by running :code:`python3 examples/bootloader/flash_bootloader.py`.

This demo runs an HTTP server on the device itself. Server will serve you a static image when you connect to it (send a GET request).

Demo
####

When you run the demo, it will print something similar to

.. code-block:: bash

    Serving at 192.168.1.193:8080

If you open this IP in the browser (eg. chrome), you will see this:

.. image:: /_static/images/components/http_server.png

if you click on the :code:`here` href, you will get a static image. For video stream,
you should check out :ref:`Script MJPEG server`.

Setup
#####

.. include::  /includes/install_from_pypi.rst

Source code
###########

.. tabs::

    .. tab:: Python

        Also `available on GitHub <https://github.com/luxonis/depthai-python/blob/main/examples/Script/script_http_server.py>`__

        .. literalinclude:: ../../../../examples/Script/script_http_server.py
           :language: python
           :linenos:

    .. tab:: C++

        Also `available on GitHub <https://github.com/luxonis/depthai-core/blob/main/examples/Script/script_http_server.cpp>`__

        .. literalinclude:: ../../../../depthai-core/examples/Script/script_http_server.cpp
           :language: cpp
           :linenos:

.. include::  /includes/footer-short.rst
