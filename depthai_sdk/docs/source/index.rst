===========
DepthAI SDK
===========

DepthAI SDK is a Python package built on top of the `depthai-python <https://github.com/luxonis/depthai-python>`__ API library that **improves
ease of use when developing apps for OAK devices**.

It abstracts DepthAI API pipeline building, different camera permutations, stream :ref:`recording <Recording>`/:ref:`replaying <Replaying>`, adds
debugging features, handles :ref:`AI model <AI models>` sourcing and decoding, does message syncing & visualization, and much more.

.. image:: https://user-images.githubusercontent.com/18037362/142909472-eae7bed1-695b-48ec-8895-76982989de11.png



Installation
------------
.. include::  install.rst

To help you understand and learn how to use the package and how the manager classes work, we created a few simple tutorials:

- :ref:`Getting started with DepthAI SDK`


We recommend that you start with the :ref:`Getting started with DepthAI SDK`

.. include::  footer-short.rst

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Home:

   self
   getting-started.rst
   oak-camera.rst

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Features

   features/ai_models.rst
   features/packets.rst

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Components:


   components/components.rst
   components/nn_component.rst
   components/camera_component.rst
   components/stereo_component.rst
