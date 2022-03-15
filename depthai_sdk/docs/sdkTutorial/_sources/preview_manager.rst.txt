===============
Preview manager
===============

``Preview manager`` is a class that is made to help you with displaying previews / streams from OAK cameras.

Getting started
---------------

``Preview manager`` works hand in hand with the ``Pipeline manager``, so before you can use Preview, you will first have to declare and initialize the ``Pipeline manager``.
But of course you will also have to import bot names to use them.

.. literalinclude:: ./examples/code_fractions/previews.py
   :language: python
   :linenos:

As you can see from the above code, we first initialized the pipeline, after that we defined sources from where the pipeline will stream and after that we connected to the device. When the device is connected,
we can declare and initialize the ``Preview manager`` and after that we can see the frames as outputs. This tutorial will only show the first frame and after that it will disconnect from the device.
If you wish to have a non-stop stream you will have to include a stoppable loop.

Example of use
--------------

.. literalinclude:: ./examples/camera_preview.py
   :language: python
   :linenos:

In the above example we added a stoppable loop (while stops when 'q' is pressed) and we also defined a few more sources.
Output of the above code should look something like this:

.. image:: /_static/images/camera_previews.png

We get frames from all defined sources.
To see more about the ``Preview manager`` check ``API``.

.. include::  footer-short.rst