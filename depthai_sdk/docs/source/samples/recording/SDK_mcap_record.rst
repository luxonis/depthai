MCAP Recording 
==============

This example showcases the use of SDK to save to MCAP file format. The MCAP file contains color as well as both left and right mono cameras and their inferred depth map.  

.. include::  /includes/blocking_behaviour.rst
    


Setup
#####

.. include::  /includes/install_from_pypi.rst


Pipeline
########

.. image:: /_static/images/pipelines/mcap_record.png
      :alt: Pipeline graph



Source Code
###########

.. tabs::

    .. tab:: Python

        Also `available on GitHub <https://github.com/luxonis/depthai/depthai_sdk/recording/mcap_record.py>`_.



        .. literalinclude:: ../../../../examples/recording/mcap_record.py
            :language: python
            :linenos:

.. include::  /includes/footer-short.rst