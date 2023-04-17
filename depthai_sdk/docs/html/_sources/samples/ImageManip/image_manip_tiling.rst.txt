ImageManip Tiling
=================

Frame tiling could be useful for eg. feeding large frame into a :ref:`NeuralNetwork` whose input size isn't as large. In such case,
you can tile the large frame into multiple smaller ones and feed smaller frames to the :ref:`NeuralNetwork`.

In this example we use 2 :ref:`ImageManip` for splitting the original :code:`1000x500` preview frame into two :code:`500x500` frames.

Demo
####

.. image:: https://user-images.githubusercontent.com/18037362/128074673-045ed4b6-ac8c-4a76-83bb-0f3dc996f7a5.png
  :alt: Tiling preview into 2 frames/tiles

Setup
#####

.. include::  /includes/install_from_pypi.rst

Source code
###########

.. tabs::

    .. tab:: Python

        Also `available on GitHub <https://github.com/luxonis/depthai-python/blob/main/examples/ImageManip/image_manip_tiling.py>`__

        .. literalinclude:: ../../../../examples/ImageManip/image_manip_tiling.py
           :language: python
           :linenos:

    .. tab:: C++

        Also `available on GitHub <https://github.com/luxonis/depthai-core/blob/main/examples/ImageManip/image_manip_tiling.cpp>`__

        .. literalinclude:: ../../../../depthai-core/examples/ImageManip/image_manip_tiling.cpp
           :language: cpp
           :linenos:

.. include::  /includes/footer-short.rst
