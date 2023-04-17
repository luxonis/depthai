Encoded bitstream (either MJPEG, H264, or H265) from the device can also be **saved directly into .mp4 container**
with no computational overhead on the host computer. See `demo here <https://github.com/luxonis/depthai-experiments/tree/master/gen2-container-encoding>`__
for more information.

**Matroska**

Besides ``ffmpeg`` and ``.mp4`` video container (which is patent encumbered), you could also use the ``mkvmerge``
(see `MKVToolNix <https://mkvtoolnix.download/doc/mkvmerge.html>`__ for GUI usage) and ``.mkv`` video container
to mux encoded stream into video file that is supported by all major video players
(eg. `VLC <https://www.videolan.org/vlc/>`__)

.. code-block::
    
    mkvmerge -o vid.mkv video.h265