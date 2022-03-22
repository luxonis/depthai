================
Encoding manager
================

``Encoding manager`` is a class that is made to help you with creating videos from OAK cameras.

Getting started
---------------

Same as ``Preview manager``, ``Encoding manager`` works hand in hand with the ``Pipeline manager``, so before you can use the encoder, you will first have to declare and initialize the ``Pipeline manager``.
Again, same as the ``Preview manager``, it is not needed to use ``Pipeline manager``, you can create the pipeline without it. The managers are created to help you and make your programing experience with ``DepthAi SDK`` easier.

.. literalinclude:: ./examples/code_fractions/encoder.py
   :language: python
   :linenos:

The most important part, when using the ``Encoding manager`` is that the encoder must be created during the pipeline creation.
So to begin, first we create a dictionary, which will contain all streams from the OAK camera as keys, and number of fps as
their values. After that we declare the ``Encoding manager`` with the dictionary, which we just declared, and the wanted path,
where the files will be stored. So to store our videos we have to give our encoder the wanted path for saving. We specify our path with ``Path(__file__)``. All the files will be stored in ``.h265`` format,
with the file name beeing the source name (so in the above example we will create color.h256).

As you can also see after we declare the pipeline and initialize it's sources, we must set ``xoutVideo`` to ``True`` instead of ``xout``.
And after connecting to the device we parse through the queues and save frames to files.


To see more about the ''Encoding manager'' check ``API``.

.. include::  footer-short.rst