# DepthAI

We are launching a CrowdSupply campaign (https://www.crowdsupply.com/luxonis/depthai) to bring the power of the Myriad X to your design.  If the campaign is successful we will be releasing the designs (hardware and software) for the following Myriad X carrier boards here on Github - to enable engineers and makers to build their own solutions, leveraging our source designs/code.

![DepthAI Models](/images/67516272-9e50d600-f65d-11e9-9343-8a8c3425c47d.png)

In addition to the designs shown above (which will be purchase-able through the campaign), we will be releasing an ordered, but not-yet-received version which integrates all 3 cameras onto a single board with a USB3C interface to the host, shown on the bottom right below.

![DepthAI Models](/images/67443015-55970f80-f5c0-11e9-83c3-2bf07a2479e3.png)

The DepthAI platform is engineered from the ground up to enable the original vision of the Myriad X - which is to allow low-power, high-efficiency vision processing including stereo depth, motion estimation, and neural inference.  Existing platforms have no MIPI interface and only have a PCIE or USB inferface and so cannot take advantage of a TON of the hardware modules built into the Myriad X which are only (meaningfully) usable with MIPI:

 - Stereo Depth 
 - Edge Detection 
 - Harris Filtering
 - Warp/De-Warp
 - Motion Estimation
 - ISP Pipeline
 - JPEG Encoding
 - H.264 and H.265 Encoding
 
 So to allow the use of these, DepthAI is engineered to allow direct MIPI connection:
 
 ![DepthAI Models](/images/67444612-10c2a700-f5c7-11e9-8018-5485c2dad580.png)
 
 And to allow this power to be integrate-able into actual products (which may require differing stereo baselines, differing interfaces, geometries, etc.) and to make hardware integration much easier (removing the need to integrate a very-fine-pitch multi-hundred-ball BGA into your design) we made a System on Module to hold the Myriad X:
 
 ![DepthAI Models](/images/67533825-9e1a0000-f688-11e9-95a7-26206fdb9a43.png)
 
 This module enables a simple and standard tolerance PCB, instead of the high-layer-count and high-density-integration (including laser vias and stacked vias) required to integrate the Myriad X directly.  And it also provides all the power supply rails, power sequencing, clock generation, and camera-support hardware on-module.
 
 This allows you to integrate the power of the Myriad X into your design with a standard 4-layer (or even 2-layer, gasp!) PCB.  And then leverage our reference hardware and software designs to get up/going super-fast (or just buy our boards directly, if they happen to fit the bill for what you need).  Which all results in getting up/running fast:
![DepthAI Models](/images/67452322-0b258b00-f5e0-11e9-843d-09c6231fb8b9.png)

So what are the reference designs (the carrier boards) which will be released should the campaign be funded?

** DepthAI | Raspberry Pi Compute Module **
![DepthAI Models](/images/67506624-a6ebe100-f64a-11e9-9f3b-12af23c2fa6c.png)
![DepthAI Models](/images/67516510-01db0380-f65e-11e9-99e0-7d635781e377.png)

# DepthAI | Raspberry Pi HAT with FFC Cameras
![DepthAI Models](/images/67524974-078d1500-f66f-11e9-9b86-cd7578f63b42.png)
![DepthAI Models](/images/67516846-92b1df00-f65e-11e9-974b-b37825192901.png)
![DepthAI Models](/images/67516891-a8270900-f65e-11e9-9ad1-3318f49396e5.png)

# DepthAI | USB3 with FFC Camera
![DepthAI Models](/images/67526422-f691d300-f671-11e9-8b11-e574e808c619.png)
![DepthAI Models](/images/67530493-745adc00-f67c-11e9-86bb-d78ba7150d16.png)
![DepthAI Models](/images/67530766-63f73100-f67d-11e9-9f9a-e7ca269832cb.png)

 
 

