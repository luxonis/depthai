============
Blob manager
============

``Blob manager`` is a class that is made to help you with downloading neural networks as MyriadX blobs.

Getting started
---------------

``Blob manager`` is very easy and straight forward to use. We declare it and pass which project we want to use as it's argument.
Manager supports all models in both ``Open Model Zoo`` and our ``Model zoo``.
By using ``configPath`` and ``zooDir`` when initializing the ``Blob manager`` you can specify a path to a custom model zoo (path can contain both already compiled blobs and model yml files),
which will compile a new blob or read an existing blob.
``Blob manager`` is able to reuse existing blobs, download models from model zoo and compile custom models based on yml config.

.. literalinclude:: ./examples/code_fractions/blob.py
   :language: python
   :linenos:

After that, the ``blob`` is stored in our variable and we can then pass it to our ``Nnet manager``, as we will see in the next tutorial, or use it in any other project that we wish.

If you wish to learn more about ``Blob manager`` check :ref:`DepthAI SDK API`.

.. include::  footer-short.rst