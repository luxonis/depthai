Code Samples
============

.. toctree::
   :hidden:
   :glob:

   ../samples/CameraComponent/*
   ../samples/mixed/*
   ../samples/IMUComponent/*
   ../samples/NNComponent/*
   ../samples/PointcloudComponent/*
   ../samples/recording/*
   ../samples/replay/*
   ../samples/StereoComponent/*
   ../samples/streaming/*
   ../samples/trigger_action/*
   ../samples/visualizer/*

Code samples are used for automated testing. They are also a great starting point for the DepthAI SDK, as different component functionalities
are presented with code.


.. rubric:: ColorCamera

- :ref:`FFC Camera Visualization` - Preview FFC Cameras
- :ref:`Camera Control` - Demonstrates RGB camera control from the host
- :ref:`Camera Preview` - Preview color, right, left and depth frames
- :ref:`Camera Control with NN` - Control camera (focus, exposure) with NN detections
- :ref:`Mono Camera Preview` - Preview mono cameras with manual 400p resolution
- :ref:`Preview All Cameras` - Preview all cameras connected to the OAK device
- :ref:`RGB and Mono Preview` - Preview RGB and mono cameras
- :ref:`Camera Rotated Preview` - Demonstrates how to rotate the camera previews

.. rubric:: Mixed
- :ref:`API Interoperability Example` - Demonstrates interoperability between the DepthAI API and the SDK
- :ref:`Car Tracking Example` - Demonstrates how to run inference on a pre-saved video
- :ref:`Collision Avoidance` - Demonstrates how to run collision avoidance
- :ref:`Speed Calculation Preview` - Demonstrates how to calculate speed of detected objects in the frame`
- :ref:`Switch Between Models` - Demonstrates how to switch between models
- :ref:`Sync Multiple Outputs` - Demonstrates how to sync multiple outputs

.. rubric:: IMU
- :ref:`IMU Demonstration` - Demonstrates how to use and display the IMU
- :ref:`IMU Rerun Demonstration` - Demonstrates how use and display the IMU in Rerun

.. rubric:: NN
- :ref:`Age-Gender Inference` - Demonstrates age-gender inference
- :ref:`Custom Decode Function` - Demonstrates custom decoding function
- :ref:`Emotion Recognition` - Demonstrates emotion recognition
- :ref:`Face Detection RGB` - Run face detection on RGB camera
- :ref:`Face Detection Mono` - Run face detection on mono camera
- :ref:`Human Pose Estimation` - Run human pose estimation inference
- :ref:`MobileNet Encoded` - Pass encoded color stream to the NN (MobileNet)
- :ref:`Neural Network Component` - Run color camera stream through NN (YoloV7)
- :ref:`Object Tracking` - Tracking objects in the frame
- :ref:`Roboflow Integration` - Demonstrates how to use Roboflow platform to train a custom model
- :ref:`Spatial Detection` - Perform spatial detection with at MobileNet model
- :ref:`Yolo SDK` - Run YoloV3 inference on the color camera stream

.. rubric:: Pointcloud
- :ref:`Pointcloud Demo` - Preview pointcloud with rerun viewer

.. rubric:: Recording
- :ref:`Encode Multiple Streams` - Demonstrates how to encode multiple (color, left, right) streams and save them to a file
- :ref:`Preview Encoder` - Record color camera stream and save it as mjpeg
- :ref:`MCAP Recording` - Record color, left, right and depth streams and save them to a MCAP
- :ref:`MCAP IMU Recording` - Record IMU and depth streams and save them to a MCAP
- :ref:`Hardcode Recording Duration` - Record color camera stream for a specified duration
- :ref:`ROSBAG Recording` - Record IMU, left, right and depth streams and save them to a ROSBAG
- :ref:`Stereo Recording` - Records disparity stream

.. rubric:: Replay
- :ref:`Object counting on images` - Count number of objects on a folder of images (cycle through images every 3 sec)
- :ref:`People Tracker on Video Replay` - Run people tracker on a pre-saved video
- :ref:`Face Detection Inference on Downloaded Image` - Run face detection on a downloaded image
- :ref:`Vehicle Detection on a Youtube Video` - Run vehicle detection on a Youtube video stream
- :ref:`Looped Replay` - Replay a pre-saved video in a loop

.. rubric:: Stereo
- :ref:`Stereo Preview` - Display WLS filtered disparity map
- :ref:`Auto IR Brightness` - Demonstrates the use of auto IR brightness function
- :ref:`Stereo Control` - Demonstrates stereo control (median filter, decimation factor, confidence threshold) from the host
- :ref:`Stereo Encoding` - Demonstrates how to encode stereo stream and visualize it

.. rubric:: Streaming
- :ref:`ROS Publishing` - Publish color, left, right and IMU streams to ROS

.. rubric:: Trigger Action
- :ref:`Custom Trigger Action` - Demonstrates how to set a custom trigger action
- :ref:`Custom Trigger` - Demonstrates how to set a custom trigger
- :ref:`Person Record` - Demonstrates how to record a person when a person is detected

.. rubric:: Visualizer
- :ref:`Visualizer Demo` - Demonstrates how to use the visualizer
- :ref:`Visualizer Callback Function` - Demonstrates how to set the visualizer callback function
