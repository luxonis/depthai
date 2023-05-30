IMUComponent
============

**IMUComponent** abstracts `IMU <https://docs.luxonis.com/projects/api/en/latest/components/nodes/imu/>`__ node and its configuration.

Usage
#####

.. code-block:: python

    from depthai_sdk import OakCamera
    from depthai_sdk.classes import IMUPacket

    with OakCamera() as oak:
        imu = oak.create_imu()
        imu.config_imu(report_rate=400, batch_report_threshold=5)

        def callback(packet: IMUPacket):
            print(packet)

        oak.callback(imu.out.main, callback=callback)
        oak.start(blocking=True)

Component outputs
#################

- :attr:`main <depthai_sdk.components.IMUComponent.Out.main>` - Main output, produces :ref:`IMUPacket`.

Reference
#########

.. autoclass:: depthai_sdk.components.IMUComponent
    :members:
    :undoc-members: