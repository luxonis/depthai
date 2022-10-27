OakCamera
=========

The **OakCamera** class abstracts:

- DepthAI API pipeline building with :ref:`Components`
- Stream :ref:`recording <Recording>` and :ref:`replaying <Replaying>`
- Debugging features (such as ``oak.show_graph()``)
- :ref:`AI model <AI models>` sourcing and decoding
- Message syncing & visualization, and much more

.. note::
    This class will be in **alpha stage** until **depthai-sdk 2.0.0**, so there will likely be some API changes.


Interoperability with DepthAI API
---------------------------------

DepthAI SDK was develop with `DepthAI API <https://docs.luxonis.com/projects/api/en/latest/>`__ interoperability in mind.
Users can access all depthai API nodes inside components, and after ``oak.build()`` also the `dai.Pipeline <https://docs.luxonis.com/projects/api/en/latest/components/pipeline/>`__
and `dai.Device <https://docs.luxonis.com/projects/api/en/latest/components/device/>`__ objects.

.. literalinclude:: ./examples/api_interop.py
   :language: python


Reference
#########

.. autoclass:: depthai_sdk.OakCamera
    :members:
    :undoc-members:

.. include::  includes/footer-short.rst