RGB Camera Control
==================

This example shows how to control the :ref:`ColorCamera`, such as exposure, sensitivity, white balance, luma/chroma denoise,
device-side crop, camera triggers, etc.

An output is a full frame along with cropped frame (``video``) which can be moved with ``WASD`` keys.

**List of all controls**:

.. code-block::

  Uses 'WASD' controls to move the crop window, 'C' to capture a still image, 'T' to
  trigger autofocus, 'IOKL,.NM' for manual exposure/focus/white-balance:
    Control:      key[dec/inc]  min..max
    exposure time:     I   O      1..33000 [us]
    sensitivity iso:   K   L    100..1600
    focus:             ,   .      0..255 [far..near]
    white balance:     N   M   1000..12000 (light color temperature K)

  To go back to auto controls:
    'E' - autoexposure
    'F' - autofocus (continuous)
    'B' - auto white-balance

  Other controls:
    '1' - AWB lock (true / false)
    '2' - AE lock (true / false)
    '3' - Select control: AWB mode
    '4' - Select control: AE compensation
    '5' - Select control: anti-banding/flicker mode
    '6' - Select control: effect mode
    '7' - Select control: brightness
    '8' - Select control: contrast
    '9' - Select control: saturation
    '0' - Select control: sharpness
    '[' - Select control: luma denoise
    ']' - Select control: chroma denoise

  For the 'Select control: ...' options, use these keys to modify the value:
    '-' or '_' to decrease
    '+' or '=' to increase

.. rubric:: Similar samples:

- :ref:`Mono Camera Control`
- :ref:`Depth Crop Control`

Demo
####

.. raw:: html

    <div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; height: auto;">
        <iframe src="https://www.youtube.com/embed/httOxe2LAkI" frameborder="0" allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe>
    </div>

Setup
#####

.. include::  /includes/install_from_pypi.rst

Source code
###########

.. tabs::

    .. tab:: Python

        Also `available on GitHub <https://github.com/luxonis/depthai-python/blob/main/examples/ColorCamera/rgb_camera_control.py>`__

        .. literalinclude:: ../../../../examples/ColorCamera/SDK_RGB_controls.py
           :language: python
           :linenos:

.. include::  /includes/footer-short.rst
