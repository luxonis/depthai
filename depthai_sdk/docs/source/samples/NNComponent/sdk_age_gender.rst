Age-Gender Inference
====================

This example showcases the usage of multi-stage neural network pipeline to make age and gender inference on a video frame. 

.. include::  /includes/blocking_behaviour.rst
    


Setup
#####

.. include::  /includes/install_from_pypi.rst

Pipeline
########

.. image:: /_static/images/pipelines/age-gender.png
      :alt: Pipeline graph


Source Code
###########

.. tabs::

    .. tab:: Python

        Also `available on GitHub <https://github.com/luxonis/depthai/depthai_sdk/NNComponent/age-gender.py>`_.

        .. literalinclude:: ../../../../examples/NNComponent/age-gender.py
            :language: python
            :linenos:

.. include::  /includes/footer-short.rst