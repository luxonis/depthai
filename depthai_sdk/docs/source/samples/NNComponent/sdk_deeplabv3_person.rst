Deeplabv3 Person Segmentation 
=============================

This example showcases the implementation of deepLabv3 person segmentation model with DepthAI SDK. 

.. include::  /includes/blocking_behavior.rst
    


Setup
#####

.. include::  /includes/install_from_pypi.rst

Pipeline
########

.. image:: /_static/images/pipelines/deeplabv3_person.png
      :alt: Pipeline graph



Source Code
###########

One thing worth noting is the resize mode option. Because inference is done on a color camera which has a 16:9 aspect ratio, and the model expects a 1:1 aspect ratio, we need
to resize the input frame to fit the model. This is done in three ways:

- letterbox - resize the frame to fit the model, and pad the rest with black pixels
- crop - crop the frame to fit the model
- stretch - stretch the frame to fit the model

More information at `Maximizing FOV <https://docs.luxonis.com/projects/api/en/latest/tutorials/maximize_fov/>`__.


.. tabs::


    .. tab:: Python

        Also `available on GitHub <https://github.com/luxonis/depthai/tree/main/depthai_sdk/examples/NNComponent/deeplabv3_person.py>`__
        
        .. literalinclude:: ../../../../examples/NNComponent/deeplabv3_person.py
            :language: python
            :linenos:

.. include::  /includes/footer-short.rst