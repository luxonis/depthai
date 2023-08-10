Person Record 
=============

This example shows how to set up a trigger with a RecordAction to record both color and disparity frames when a condition is met.

.. include::  /includes/blocking_behavior.rst
    

Setup
#####

.. include::  /includes/install_from_pypi.rst

Pipeline
########

.. image:: /_static/images/pipelines/person_record.png
      :alt: Pipeline graph



Source Code
###########

.. tabs::

    .. tab:: Python

        Also `available on GitHub <https://github.com/luxonis/depthai/tree/main/depthai_sdk/examples/trigger_action/person_record.py>`_.

        .. literalinclude:: ../../../../examples/trigger_action/person_record.py
            :language: python
            :linenos:

.. include::  /includes/footer-short.rst