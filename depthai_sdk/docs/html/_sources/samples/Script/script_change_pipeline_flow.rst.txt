Script change pipeline flow
===========================

This example shows how you can change the flow of data inside your pipeline in runtime using the :ref:`Script` node. In this example, we send a message from
the host to choose whether we want Script node to forwards color frame to the :ref:`MobileNetDetectionNetwork`.

Demo
####

.. image:: https://user-images.githubusercontent.com/18037362/187734814-df3b46c9-5e04-4a9d-bf6f-d738b40b4421.gif

Pipeline Graph
##############

.. image:: https://user-images.githubusercontent.com/18037362/187736249-db7ff175-fcea-4d4e-b567-f99087bd82ee.png

Setup
#####

.. include::  /includes/install_from_pypi.rst

Source code
###########

.. tabs::

    .. tab:: Python

        Also `available on GitHub <https://github.com/luxonis/depthai-python/blob/main/examples/Script/script_change_pipeline_flow.py>`__

        .. literalinclude:: ../../../../examples/Script/script_change_pipeline_flow.py
           :language: python
           :linenos:

    .. tab:: C++

        Also `available on GitHub <https://github.com/luxonis/depthai-core/blob/main/examples/Script/script_change_pipeline_flow.cpp>`__

        .. literalinclude:: ../../../../depthai-core/examples/Script/script_change_pipeline_flow.cpp
           :language: cpp
           :linenos:

.. include::  /includes/footer-short.rst
