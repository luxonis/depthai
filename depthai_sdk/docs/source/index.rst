.. Luxonis Docs documentation master file, created by
   sphinx-quickstart on Tue Nov  3 14:34:56 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

DepthAI SDK documentation
=========================

DepthAI SDK is a Python package, containing convenience classes and functions that help in most common tasks while using DepthAI API.

To know more about the DepthAI itself, visit `our documentation <https://docs.luxonis.com>`__ or `our webpage <https://luxonis.com>`__

Installation
------------

To install this package, run the following command in your terminal window

.. code-block:: bash

   pip install depthai-sdk

.. warning::

   If you're using Raspberry Pi, providing a Pi Wheels extra package url can significantly speed up the instalation process by providing prebuilt binaries for OpenCV

   .. code-block:: bash

      pip install --extra-index-url https://www.piwheels.org/simple/ depthai-sdk


API
---

See :ref:`DepthAI SDK API`

.. toctree::
   :maxdepth: 0
   :hidden:
   :caption: Content:

   api.rst