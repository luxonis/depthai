Interactive Warp Mesh
=====================

This example shows usage of :ref:`Warp` node to warp the input image frame. It let's you interactively change the mesh points to warp the image. After changing the points,
**user has to press** ``r`` to restart the pipeline and apply the changes.

User-defined arguments:

- ``--mesh_dims`` - Mesh dimensions (default: ``4x4``).
- ``--resolution`` - Resolution of the input image (default: ``512x512``). Width must be divisible by 16.
- ``--random`` - To generate random mesh points (disabled by default).

Originally developed by `geaxgx <https://github.com/geaxgx>`__.

Setup
#####

.. include::  /includes/install_from_pypi.rst

Demo
####

.. figure:: https://user-images.githubusercontent.com/18037362/214605914-87cf0404-2d89-478f-9062-2dfb4baa6512.png

    Original and warped image


Source code
###########

.. tabs::

    .. tab:: Python

        Also `available on GitHub <https://github.com/luxonis/depthai-python/blob/main/examples/Warp/warp_mesh_interactive.py>`__

        .. literalinclude:: ../../../../examples/Warp/warp_mesh_interactive.py
           :language: python
           :linenos:

    .. tab:: C++

        WIP

.. include::  /includes/footer-short.rst
