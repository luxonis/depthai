Looped Replay
=============


This example shows how to run replay in a loop. This means the device won't close when the replay file ends. 


.. include::  /includes/blocking_behavior.rst
    


Setup
#####

.. include::  /includes/install_from_pypi.rst

Pipeline
########

.. image:: /_static/images/pipelines/looped_replay.png
      :alt: Pipeline graph



Source Code
###########

.. tabs::

    .. tab:: Python

        Also `available on GitHub <https://github.com/luxonis/depthai/tree/main/depthai_sdk/examples/replay/looped_replay.py>`_.


        .. literalinclude:: ../../../../examples/replay/looped-replay.py
            :language: python
            :linenos:

.. include::  /includes/footer-short.rst