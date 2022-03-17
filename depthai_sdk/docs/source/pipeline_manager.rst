================
Pipeline manager
================

Pipeline is the first class that we will learn how to use as it is the one that is goes hand in hand with every other class.
It is created with the purpose to help you with creating and setting up processing pipelines. In this tutorial bellow we will see and learn how to declare and use them.


Getting started
---------------

Before we begin we must first import ``pipeline_manager`` from ``DepthAI_SDK``. After that we will initialize the pipeline and define its sources.

.. literalinclude:: ./examples/code_fractions/pipeline.py
   :language: python
   :linenos:

We successfully created and initialized the pipeline. If everything was setup correctly, you should receive a message in your terminal, that will inform you that the connecting was successful.

.. image:: /_static/images/connecting_message.png

But the above code currently one has one source as it's stream. We can initialize more sources in one pipeline.

.. literalinclude:: ./examples/code_fractions/pipeline_more.py
   :language: python
   :linenos:

We now declared more then one source in the pipeline. To fully use the pipeline, you can use it with ``Preview manager`` to see the streams or ``Encoding manager`` to save streams to files.
As you can see above we also added another argument to the color camera stream, called ``previewSize`` which will resize the stream to wanted ratio (height x width). All sources have many more arguments,
``xout`` will help us in the next tutorial where we will learn about the ``Preview manager``. To see all arguments that the streams can contain and everything else about the ``Pipeline manager``
check ``API``.

.. include::  footer-short.rst