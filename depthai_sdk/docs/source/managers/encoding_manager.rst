================
Encoding manager
================

:obj:`EncodingManager` is a class that is made to help you with creating videos from OAK cameras.

Getting started
---------------

Same as :obj:`PreviewManager`, :obj:`EncodingManager` works hand in hand with the :obj:`PipelineManager`, so before you can use the encoder, you will first have to declare and initialize the :obj:`PipelineManager`.
Again, same as the :obj:`PreviewManager`, it is not needed to use :obj:`PipelineManager`, you can create the pipeline without it. The managers are created to help you and make your programing experience with ``DepthAi SDK`` easier.

.. literalinclude:: ../examples/code_fractions/encoder.py
   :language: python
   :linenos:

The most important part, when using the :obj:`EncodingManager` is that the encoder must be created during the pipeline creation.
So to begin, first we create a dictionary, which will contain all streams from the OAK camera as keys, and number of fps as
their values. After that we declare the :obj:`EncodingManager` with the dictionary, which we just declared, and the wanted path,
where the files will be stored. So to store our videos we have to give our encoder the wanted path for saving. We specify our path with ``Path(__file__)``. All the files will be stored in ``.h265`` format,
with the file name beeing the source name (so in the above example we will create color.h256).

As you can also see after we declare the pipeline and initialize it's sources, we must set ``xoutVideo`` to ``True`` instead of ``xout``.
And after connecting to the device we parse through the queues and save frames to files.


To see more about the :obj:`EncodingManager` check ``API``.

.. include::  footer-short.rst