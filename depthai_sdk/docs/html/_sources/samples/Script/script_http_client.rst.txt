Script HTTP client
==================

.. note::
    This example can only run on `OAK POE devices <https://docs.luxonis.com/projects/hardware/en/latest/#poe-designs>`__. You need **bootloader** on/above version **0.0.15**. You can flash bootloader by running :code:`python3 examples/bootloader/flash_bootloader.py`.

This example will send an `HTTP request <https://en.wikipedia.org/wiki/Hypertext_Transfer_Protocol>`__ to the http://api.ipify.org and print your public IP.

Demo
####

.. code-block:: bash

    ~/depthai-python/examples/Script$ python3 script_http_client.py
    Sending http GET request...
    200 OK
    Public IP: b'123.123.123.123'

Setup
#####

.. include::  /includes/install_from_pypi.rst

Source code
###########

.. tabs::

    .. tab:: Python

        Also `available on GitHub <https://github.com/luxonis/depthai-python/blob/main/examples/Script/script_http_client.py>`__

        .. literalinclude:: ../../../../examples/Script/script_http_client.py
           :language: python
           :linenos:

    .. tab:: C++

        Also `available on GitHub <https://github.com/luxonis/depthai-core/blob/main/examples/Script/script_http_client.cpp>`__

        .. literalinclude:: ../../../../depthai-core/examples/Script/script_http_client.cpp
           :language: cpp
           :linenos:

.. include::  /includes/footer-short.rst
