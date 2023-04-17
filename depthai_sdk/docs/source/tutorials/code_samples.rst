Code Samples
============

.. toctree::
   :hidden:
   :glob:

   ../samples/ColorCamera/*
   ../samples/mixed/*
   

Code samples are used for automated testing. They are also a great starting point for the DepthAI API, as different node functionalities
are presented with code.


.. rubric:: ColorCamera- Demonstrates how to control the RGB camera (crop, focus, exposure, sensitivity) from the host


- :ref:`Auto Exposure on ROI` - Demonstrates how to use auto exposure based on the selected ROI
- :ref:`RGB Camera Control` 
.. rubric:: Mixed

- :ref:`RGB Encoding & MobilenetSSD` - Runs MobileNetSSD on RGB frames and encodes FUll-HD RGB into :code:`.h265` and saves it on the host
- :ref:`RGB Encoding & Mono & MobilenetSSD` - Runs MobileNetSSD on mono frames and displays detections on the frame + encodes RGB to :code:`.h265`
