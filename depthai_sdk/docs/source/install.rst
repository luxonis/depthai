To install this package, run the following command in your terminal window

.. code-block:: bash

   python3 -m pip install depthai-sdk

.. warning::

   If you're using Raspberry Pi, providing a Pi Wheels extra package url can significantly speed up the installation process by providing prebuilt binaries for OpenCV

   .. code-block:: bash

      python3 -m pip install --extra-index-url https://www.piwheels.org/simple/ depthai-sdk