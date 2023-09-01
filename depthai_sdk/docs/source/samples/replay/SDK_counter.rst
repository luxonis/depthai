Object counting on images
=========================

This example cycles through a folder of images and counts the number of objects (people in our case) in each image. It displays the count number on the top of the image. It cycles through
each image every 3 seconds, but you can change that with:

.. code-block:: bash

    with OakCamera('path/to/folder') as oak:
      oak.replay.set_fps(0.5) # For switching cycling through image every 2 seconds
      # ...

.. include::  /includes/blocking_behavior.rst

Demo
####
.. image:: /_static/images/demos/sdk_counter.gif
      :alt: Counter demo

Setup
#####

.. include::  /includes/install_from_pypi.rst

Pipeline
########

.. image:: /_static/images/pipelines/counter.png
      :alt: Pipeline graph



Source Code
###########

.. tabs::

    .. tab:: Python

        Also `available on GitHub <https://github.com/luxonis/depthai/tree/main/depthai_sdk/examples/replay/counter.py>`_.

        .. literalinclude:: ../../../../examples/replay/counter.py
            :language: python
            :linenos:

.. include::  /includes/footer-short.rst