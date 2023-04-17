Script NNData example
=====================

This example shows how to create a :ref:`NNData` message inside the :ref:`Script` node and then
send it to the host (where it gets printed to the console).

Demo
####

.. code-block:: bash

    ~/depthai-python/examples/Script$ python3 script_nndata_datatype.py
    Names of layers: ['fp16', 'uint8']
    NNData size: 13
    FP16 values: [1.0, 1.2001953125, 3.900390625, 5.5]
    UINT8 values: [6, 9, 4, 2, 0]

Setup
#####

.. include::  /includes/install_from_pypi.rst

Source code
###########

.. tabs::

    .. tab:: Python

        Also `available on GitHub <https://github.com/luxonis/depthai-python/blob/main/examples/Script/script_nndata_example.py>`__

        .. literalinclude:: ../../../../examples/Script/script_nndata_example.py
           :language: python
           :linenos:

    .. tab:: C++

        Also `available on GitHub <https://github.com/luxonis/depthai-core/blob/main/examples/Script/script_nndata_example.cpp>`__

        .. literalinclude:: ../../../../depthai-core/examples/Script/script_nndata_example.cpp
           :language: cpp
           :linenos:

.. include::  /includes/footer-short.rst
