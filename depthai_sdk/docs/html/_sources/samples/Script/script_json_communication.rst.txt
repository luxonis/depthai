Script JSON communication
=========================

This example shows how Script node can communicate with the outside world (host computer) using JSON serialization
by sending a :ref:`Buffer` message. Similarly, you could use :ref:`SPIIn` and :ref:`SPIOut` to send JSON between MCU and Script node via SPI.

**How would this be useful:**

2-way communication between Script node and host/MCU could be used to eg. alter the flow of an application.

For example, we might want to
use 2 different NN models on the device, one until noon and the other from noon till midnight. Host (eg. RPi) could check the current time
and when it's noon, it would send a simple message to the Script node which would start forwarding all frames to the other NeuralNetwork node
(which has different NN model).

**What it does:**

Host creates a dictionary, serializes it, sends it to the Script node. Script node receives the Buffer message, deserializes the dictionary,
changes values a bit, serializes the dictionary again and sends it to the host, which deserializes the changed dictionary and prints the new values.

Demo
####

.. code-block:: bash

    ~/depthai-python/examples/Script$ python3 script_json_communication.py
    dict {'one': 1, 'foo': 'bar'}
    [14442C1041B7EFD000] [3.496] [Script(1)] [warning] Original: {'one': 1, 'foo': 'bar'}
    [14442C1041B7EFD000] [3.496] [Script(1)] [warning] Changed: {'one': 2, 'foo': 'baz'}
    changedDict {'one': 2, 'foo': 'baz'}

Setup
#####

.. include::  /includes/install_from_pypi.rst

Source code
###########

.. tabs::

    .. tab:: Python

        Also `available on GitHub <https://github.com/luxonis/depthai-python/blob/main/examples/Script/script_json_communication.py>`__

        .. literalinclude:: ../../../../examples/Script/script_json_communication.py
           :language: python
           :linenos:

    .. tab:: C++

        Also `available on GitHub <https://github.com/luxonis/depthai-core/blob/main/examples/Script/script_json_communication.cpp>`__

        .. literalinclude:: ../../../../depthai-core/examples/Script/script_json_communication.cpp
           :language: cpp
           :linenos:

.. include::  /includes/footer-short.rst
