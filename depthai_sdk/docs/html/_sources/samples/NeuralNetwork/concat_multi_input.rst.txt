Multi-Input Frame Concationation
================================

Example concatenates all 3 inputs with a simple custom model created with PyTorch (`link here <https://github.com/luxonis/depthai-experiments/blob/master/gen2-custom-models/generate_model/pytorch_concat.py>`__,
`tutorial here <https://docs.luxonis.com/en/latest/pages/tutorials/creating-custom-nn-models/>`__).
It uses :ref:`NeuralNetwork`'s multiple input feature and links all 3 camera streams directly to the NeuralNetwork node.

Demo
####

.. image:: https://user-images.githubusercontent.com/18037362/134209980-09c6e2f9-8a26-45d5-a6ad-c31d9e2816e1.png

Setup
#####

.. include::  /includes/install_from_pypi.rst

Source code
###########

.. tabs::

    .. tab:: Python

        Also `available on GitHub <https://github.com/luxonis/depthai-python/blob/main/examples/NeuralNetwork/concat_multi_input.py>`__

        .. literalinclude:: ../../../../examples/NeuralNetwork/concat_multi_input.py
           :language: python
           :linenos:

    .. tab:: C++

        Also `available on GitHub <https://github.com/luxonis/depthai-core/tree/main/examples/NeuralNetwork/concat_multiple_input.cpp>`__

        .. literalinclude:: ../../../../depthai-core/examples/NeuralNetwork/concat_multi_input.cpp
           :language: cpp
           :linenos:

.. include::  /includes/footer-short.rst
