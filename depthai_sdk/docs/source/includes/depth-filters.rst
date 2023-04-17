.. tabs::

    .. tab:: Median

        This is a non-edge preserving `Median filter <https://en.wikipedia.org/wiki/Median_filter>`__, which can be used
        to reduce noise and smoothen the depth map. Median filter is implemented in hardware, so it's the fastest filter.

        .. doxygenenum:: dai::MedianFilter
            :project: depthai-core
            :no-link:

    .. tab:: Speckle

        **Speckle Filter** is used to reduce the speckle noise. Speckle noise is a region with huge
        variance between neighboring disparity/depth pixels, and speckle filter tries to filter this region.

        .. doxygenstruct:: dai::RawStereoDepthConfig::PostProcessing::SpeckleFilter
            :project: depthai-core
            :no-link:
            :members:

    .. tab:: Temporal

        **Temporal Filter** is intended to improve the depth data persistency by manipulating per-pixel
        values based on previous frames. The filter performs a single pass on the data, adjusting the depth
        values while also updating the tracking history. In cases where the pixel data is missing or invalid,
        the filter uses a user-defined persistency mode to decide whether the missing value should be
        rectified with stored data. Note that due to its reliance on historic data the filter may introduce
        visible blurring/smearing artifacts, and therefore is best-suited for static scenes.

        .. doxygenstruct:: dai::RawStereoDepthConfig::PostProcessing::TemporalFilter
            :project: depthai-core
            :no-link:
            :members:

    .. tab:: Spatial

        **Spatial Edge-Preserving Filter** will fill invalid depth pixels with valid neighboring depth pixels.
        It performs a series of 1D horizontal and vertical passes or iterations, to enhance the smoothness of
        the reconstructed data. It is based on `this research paper <https://www.inf.ufrgs.br/~eslgastal/DomainTransform/>`__.

        .. doxygenstruct:: dai::RawStereoDepthConfig::PostProcessing::SpatialFilter
            :project: depthai-core
            :no-link:
            :members:

    .. tab:: Threshold

        **Threshold Filter** filters out all disparity/depth pixels outside the configured min/max
        threshold values.

        .. autoclass:: depthai.RawStereoDepthConfig.PostProcessing.ThresholdFilter
            :members:
            :inherited-members:
            :noindex:

    .. tab:: Decimation

        **Decimation Filter** will sub-samples the depth map, which means it reduces the depth scene complexity and allows
        other filters to run faster. Setting :code:`decimationFactor` to 2 will downscale 1280x800 depth map to 640x400.

        .. doxygenstruct:: dai::RawStereoDepthConfig::PostProcessing::DecimationFilter
            :members:
            :project: depthai-core
            :no-link: