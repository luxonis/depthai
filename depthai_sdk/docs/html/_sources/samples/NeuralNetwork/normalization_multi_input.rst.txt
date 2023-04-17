Frame Normalization
===================

This example shows how you can normalize a frame before sending it to another neural network. Many neural network models
require frames with RGB values (pixels) in range between :code:`-0.5` to :code:`0.5`. :ref:`ColorCamera`'s preview outputs
values between :code:`0` and :code:`255`. Simple custom model, created with PyTorch (`link here <https://github.com/luxonis/depthai-experiments/blob/master/gen2-custom-models/generate_model/pytorch_normalize.py>`__, `tutorial here <https://docs.luxonis.com/en/latest/pages/tutorials/creating-custom-nn-models/>`__),
allows users to specify mean and scale factors that will be applied to all frame values (pixels).

.. math::

   output = (input - mean) / scale

.. image:: /_static/images/examples/normalize_model.png

On the host, values are converted back to :code:`0`-:code:`255`, so they can be displayed by OpenCV.

.. note::
    This is just a demo, for normalization you should use OpenVINO's `model optimizer <https://docs.openvinotoolkit.org/latest/openvino_docs_MO_DG_prepare_model_convert_model_Converting_Model_General.html>`__ arguments :code:`--mean_values` and :code:`--scale_values`.

Setup
#####

.. include::  /includes/install_from_pypi.rst

Source code
###########

.. tabs::

    .. tab:: Python

        Also `available on GitHub <https://github.com/luxonis/depthai-python/blob/main/examples/NeuralNetwork/normalization_multi_input.py>`__

        .. literalinclude:: ../../../../examples/NeuralNetwork/normalization_multi_input.py
           :language: python
           :linenos:

    .. tab:: C++

        Also `available on GitHub <https://github.com/luxonis/depthai-core/tree/main/examples/NeuralNetwork/normalization_multiple_input.cpp>`__

        .. literalinclude:: ../../../../depthai-core/examples/NeuralNetwork/normalization_multi_input.cpp
           :language: cpp
           :linenos:

.. include::  /includes/footer-short.rst
