Switch Between Models
=====================

This example shows how to switch between models on the fly. It uses script node to alter pipeline flow (either to use the yolo model or the mobilenet model).



Setup
#####

.. include::  /includes/install_from_pypi.rst

Pipeline
########

.. image:: /_static/images/pipelines/switch_between_models.png
      :alt: Pipeline graph



Source Code
###########

.. tabs::

    .. tab:: Python

        Also `available on GitHub <https://github.com/luxonis/depthai/tree/main/depthai_sdk/examples/mixed/switch_between_models.py>`_.

        .. literalinclude:: ../../../../examples/mixed/switch_between_models.py
            :language: python
            :linenos:

.. include::  /includes/footer-short.rst