================
Encoding manager
================

``Encoding manager`` is a class that is made to help you with creating videos from OAK cameras.

Getting started
---------------

Same as ``Preview manager``, ``Encoding manager`` works hand in hand with the ``Pipeline manager``, so before you can use the encoder, you will first have to declare and initialize the ``Pipeline manager``.

.. literalinclude:: ./examples/code_fractions/encoder.py
   :language: python
   :linenos:

The most important part, when using the ``Encoding manager`` is that the encoder must be initialized before the pipeline.
So to begin, first we create a dictionary, which will contain all streams from the OAK camera as keys, and number of fps as
their values. After that we declare the ``Encoding manager`` with the dictionary, which we just declared, and the wanted path,
where the files will be stored. If we send in ``Path("")``, the files will be stored next to the program.

As you can also see after we declare the pipeline and initialize it's sources, we must set ``xoutVideo`` to ``True`` instead of ``xout``.
And after connecting to the device we parse through the queues and save frames to files.

To save more than just the first frame, we must add the queue parser in a loop.

.. literalinclude:: ./examples/encoder_preview.py
   :language: python
   :linenos:


To see more about the ''Encoding manager'' check ``API``.

.. include::  footer-short.rst