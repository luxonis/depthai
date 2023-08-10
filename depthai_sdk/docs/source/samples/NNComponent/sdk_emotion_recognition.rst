Emotion Recognition
===================

This example showcases the implementation of two stage neural network pipeline, where the first stage is a face detection network, and the second stage is an emotion recognition model.

.. include::  /includes/blocking_behavior.rst
    
Demo
####
.. image:: /_static/images/demos/sdk_emotion_recognition.gif
      :alt: Emotion Recognition Demo

Setup
#####

.. include::  /includes/install_from_pypi.rst

Pipeline
########

.. image:: /_static/images/pipelines/emotion_recognition.png
      :alt: Pipeline graph



Source Code
###########

.. tabs::

    .. tab:: Python

        Also `available on GitHub <https://github.com/luxonis/depthai/tree/main/depthai_sdk/examples/NNComponent/emotion-recognition.py>`_.

        .. literalinclude:: ../../../../examples/NNComponent/emotion-recognition.py
            :language: python
            :linenos:

.. include::  /includes/footer-short.rst