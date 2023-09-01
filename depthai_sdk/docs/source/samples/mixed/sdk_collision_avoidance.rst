Collision Avoidance
===================

This example shows how to set up a depth based collision avoidance system for proximity. This can be used with supervised robotic operation where the goal is to 
limit the robot's speed when a person is detected in front of it.


.. include::  /includes/blocking_behavior.rst
    
Demo
####
.. image:: /_static/images/demos/sdk_collision_avoidance.gif
      :alt: Collision Avoidance Demo
Setup
#####

.. include::  /includes/install_from_pypi.rst

Pipeline
########

.. image:: /_static/images/pipelines/collision_avoidance.png
      :alt: Pipeline graph



Source Code
###########

.. tabs::

    .. tab:: Python

        Also `available on GitHub <https://github.com/luxonis/depthai/tree/main/depthai_sdk/examples/mixed/collision_avoidance.py>`_.

        .. literalinclude:: ../../../../examples/mixed/collision_avoidance.py
            :language: python
            :linenos:

.. include::  /includes/footer-short.rst