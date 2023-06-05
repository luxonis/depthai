Automatic IR power control
==========================

.. note:: This feature is only available on OAK devices with IR lights.

.. note:: This feature is experimental, please report any issues you encounter
          to the Luxonis team.

**Automatic IR power control** is a feature that allows the device to automatically
adjust the IR power based on the scene. This is useful for applications where
the scene is not always the same, for example when the camera is used in an
outdoor environment.

To enable automatic IR power control, you need to use :meth:`auto_ir <depthai_sdk.StereoComponent.auto_ir>` method
that accepts two parameters:

- ``auto_mode`` - ``True`` to enable automatic IR power control, ``False`` to disable it.
- ``continuous_mode`` - ``True`` to enable continuous mode, ``False`` otherwise. Requires ``auto_mode`` to be enabled.

When **automatic mode** is enabled, the device will automatically adjust the IR power after the startup.
The disparity map will be analyzed with different dot projector and illumination settings,
and once the best settings are found, the device will use them for the rest of the session.
The whole process takes around **25 seconds**.

If **continuous mode** is enabled, the device will continue to search for better settings.
In case the scene changes and disparity map quality drops below a certain threshold,
the device will automatically adjust the IR power again.

Usage
-----

The following example shows how to enable automatic IR power control in continuous mode:

.. literalinclude:: ../../../examples/StereoComponent/stereo_auto_ir.py
   :language: python