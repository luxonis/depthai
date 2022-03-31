============
Nnet manager
============

:obj:`NNetManager` is a class that is made to help you with setting up neural networks (NN). It is also responsible for all NN related functionalities.
It’s capable of creating appropriate nodes and connections, decoding neural network output automatically or by using external handler file.

Getting started
---------------

To get started we first have to know some things that the manager offers. Firstly the manager is responsible for running our NN, which means
that our manager will need a blob to work with. We can pass the blob to our :obj:`NNetManager` either with our :obj:`BlobManager` or we can pass the blob directly from the  ``blobconverter`` module.
We now have our blob, now we need to declare our pipeline through which our NN will be receiving data. This step is best done with the help of
:obj:`PipelineManager`, as the manager already contains methods for NN nodes (``addNN`` and ``setNnManager`` methods). After initializing all of that we are almost done.
For convenience we can use :obj:`PreviewManager` to parse our frames, but this step is not needed as te :obj:`NNetManager` is able to parse raw frames.
Bellow you can see 2 projects that use the :obj:`NNetManager`. Every major step is also commented for better understanding.

Face detection
--------------

.. literalinclude:: ../examples/face_detection_color.py
   :language: python
   :linenos:

In this above example we will use all classes that we learned before and run the face detection project.
First we define the pipeline and initialize the streams. After that we load in our blob (``face-detection-retail-0004``) and send it in to our :obj:`NNetManager`.
Every project has its own ``inputSize`` (desired NN input size, which should match input size defined in the network itself (width, height)) the  and ``familyName`` (supported NN types / family names are ``YOLO`` and ``nobilenet``).
After all that is initialized, we add our neural network to our pipeline and connect to our device. In our device we set our ``Previews``, to see our stream and create our stream queues.
Like every other class that we covered, we need a loop, that will keep our project running, and in our loop, we get our frames, use our neural network to draw over our frames, and then we show them on the stream.

Outputs of our above program should look like this:

.. image:: ./_static/images/face_detection.png
   :width: 250
.. image:: ./_static/images/no_face_detection.png
   :width: 250

If our face is shown, our neural network detects it, but if we cover it, our neural network will not detect it.

Mobile net
----------

.. literalinclude:: ../examples/mobile_net_ssd.py
   :language: python
   :linenos:

This example shows how to use the ``MobileNetSSD`` project. The code should be almost the same as the one that we used in the above example, with the only difference being
the blob. In this example we load the ``mobilenet-ssd`` blob and pass it to our neural network.

If you wish to learn more about :obj:`NNetManager` check :ref:`DepthAI SDK API`.

.. include::  footer-short.rst