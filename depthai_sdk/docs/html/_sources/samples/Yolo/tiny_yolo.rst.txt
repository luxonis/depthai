RGB & Tiny YOLO
===============

This example shows how to run YOLO on the RGB input frame, and how to display both the RGB
preview and the metadata results from the YOLO model on the preview. Decoding is done on the `RVC <https://docs.luxonis.com/projects/hardware/en/latest/pages/rvc/rvc2.html#rvc2>`__
instead on the host computer.

Configurable, network dependent parameters are required for correct decoding:

- **setNumClasses** - number of YOLO classes
- **setCoordinateSize** - size of coordinate
- **setAnchors** - yolo anchors
- **setAnchorMasks** - anchorMasks26, anchorMasks13 (anchorMasks52 - additionally for full YOLOv4)
- **setIouThreshold** - intersection over union threshold
- **setConfidenceThreshold** - confidence threshold above which objects are detected

**By default, Tiny YOLOv4 is used**. You can add :code:`yolo3` as a CMD argument to use Tiny YOLOv3.

Demo
####

.. raw:: html

    <div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; height: auto;">
        <iframe src="https://www.youtube.com/embed/l4jDLs9d8GI" frameborder="0" allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe>
    </div>

Setup
#####

.. include::  /includes/install_from_pypi.rst

.. include:: /includes/install_req.rst

Source code
###########

.. tabs::

    .. tab:: Python

        Also `available on GitHub <https://github.com/luxonis/depthai-python/blob/main/examples/Yolo/tiny_yolo.py>`__

        .. literalinclude:: ../../../../examples/Yolo/tiny_yolo.py
           :language: python
           :linenos:

    .. tab:: C++

        Also `available on GitHub <https://github.com/luxonis/depthai-core/blob/main/examples/Yolo/tiny_yolo.cpp>`__

        .. literalinclude:: ../../../../depthai-core/examples/Yolo/tiny_yolo.cpp
           :language: cpp
           :linenos:

.. include::  /includes/footer-short.rst
