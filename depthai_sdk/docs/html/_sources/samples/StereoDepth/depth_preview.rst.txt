Depth Preview
=============

This example shows how to set the SGBM (semi-global-matching) disparity-depth node, connects
over XLink to transfer the results to the host real-time, and displays the depth map in OpenCV.
Note that disparity is used in this case, as it colorizes in a more intuitive way.
Below is also a preview of using different median filters side-by-side on a depth image.
There are 3 depth modes which you can select inside the code:

#. `lr_check`: used for better occlusion handling. For more information `click here <https://docs.luxonis.com/en/latest/pages/faq/#left-right-check-depth-mode>`__
#. `extended_disparity`: suitable for short range objects. For more information `click here <https://docs.luxonis.com/en/latest/pages/faq/#extended-disparity-depth-mode>`__
#. `subpixel`: suitable for long range. For more information `click here <https://docs.luxonis.com/en/latest/pages/faq/#subpixel-disparity-depth-mode>`__

.. rubric:: Similar samples:

- :ref:`RGB Preview`
- :ref:`Mono Preview`
- :ref:`Stereo Depth Video`

Demo
####

.. raw:: html

    <div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; height: auto;">
        <iframe src="https://www.youtube.com/embed/GYw_iJrnGO4" frameborder="0" allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe>
    </div>

**Filtering depth using median filter**

.. raw:: html

    <div style="position: relative; padding-top: 10px;padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; height: auto;">
        <img src="https://artifacts.luxonis.com/artifactory/luxonis-depthai-data-local/images/depth-demo-filters.png" frameborder="0" allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;">
    </div>

Setup
#####

.. include::  /includes/install_from_pypi.rst

Source code
###########

.. tabs::

    .. tab:: Python

        Also `available on GitHub <https://github.com/luxonis/depthai-python/blob/main/examples/StereoDepth/depth_preview.py>`__

        .. literalinclude:: ../../../../examples/StereoDepth/depth_preview.py
           :language: python
           :linenos:

    .. tab:: C++

        Also `available on GitHub <https://github.com/luxonis/depthai-core/blob/main/examples/StereoDepth/depth_preview.cpp>`__

        .. literalinclude:: ../../../../depthai-core/examples/StereoDepth/depth_preview.cpp
           :language: cpp
           :linenos:

.. include::  /includes/footer-short.rst
