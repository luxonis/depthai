# DepthAI

UPDATE: Our Crowd Supply campaign is live!  https://www.crowdsupply.com/luxonis/depthai

We are launching a CrowdSupply campaign (https://www.crowdsupply.com/luxonis/depthai) to bring the power of the Myriad X to your design.  If the campaign is successful we will be releasing the designs (hardware and software) for the following Myriad X carrier boards here on Github - to enable engineers and makers to build their own solutions, leveraging our source designs/code.

![DepthAI Models](/images/overall-summary.png)

In addition to the designs shown above (which will be purchase-able through the campaign), we will be releasing an ordered, but not-yet-received version which integrates all 3 cameras onto a single board with a USB3C interface to the host, shown on the bottom right below.

![DepthAI Models](/images/67443015-55970f80-f5c0-11e9-83c3-2bf07a2479e3.png)

### The Why of DepthAI

The Myriad X is WAY faster and a TON of features are only accessible when cameras are connected directly to it over MIPI.  Take object detection below as an example:
![DepthAI Models](/images/67452420-66577d80-f5e0-11e9-9e32-8de8ff6da9d0.png)

So that's 25FPS instead of 8FPS... and that frame-rate is achieved with no host processing AT ALL!  (Technically, you can run DepthAI with no host at all - but often you'll want a host so that other business-logic code can be run based on the output of DepthAI... like 'pick only the ripe strawberries')

There's more to it than just being faster though... the DepthAI platform is engineered from the ground up to enable the original vision of the Myriad X - which is to allow low-power, high-efficiency vision processing including stereo depth, motion estimation, and neural inference.  Existing platforms have no MIPI interface and only have a PCIE or USB inferface and so cannot take advantage of a TON of the hardware modules built into the Myriad X which are only (meaningfully) usable with MIPI:

 - Stereo Depth 
 - Edge Detection 
 - Harris Filtering
 - Warp/De-Warp
 - Motion Estimation
 - ISP Pipeline
 - JPEG Encoding
 - H.264 and H.265 Encoding
 
 ### The How and What of DepthAI:
 
 So to allow the use of these, DepthAI is engineered to allow direct MIPI connection:
 
 ![DepthAI Models](/images/67444612-10c2a700-f5c7-11e9-8018-5485c2dad580.png)
 
 And to allow this power to be integrate-able into actual products (which may require differing stereo baselines, differing interfaces, geometries, etc.) and to make hardware integration much easier (removing the need to integrate a very-fine-pitch multi-hundred-ball BGA into your design) we made a System on Module to hold the Myriad X:
 
 ![DepthAI Models](/images/1099overview.png)
 
 This module enables a simple and standard tolerance PCB, instead of the high-layer-count and high-density-integration (including laser vias and stacked vias) required to integrate the Myriad X directly.  And it also provides all the power supply rails, power sequencing, clock generation, and camera-support hardware on-module.
 
 This allows you to integrate the power of the Myriad X into your design with a standard 4-layer (or even 2-layer, gasp!) PCB.  And then leverage our reference hardware and software designs to get up/going super-fast (or just buy our boards directly, if they happen to fit the bill for what you need).  Which all results in getting up/running fast:
![DepthAI Models](/images/67452322-0b258b00-f5e0-11e9-843d-09c6231fb8b9.png)

So what are the reference designs (the carrier boards) which will be released should the campaign be funded?

#### DepthAI | Raspberry Pi Compute Module
![DepthAI Models](/images/67506624-a6ebe100-f64a-11e9-9f3b-12af23c2fa6c.png)
![DepthAI Models](/images/67516510-01db0380-f65e-11e9-99e0-7d635781e377.png)

This design allows you to have the Myriad X, all the cameras, and the Raspberry Pi on one board.  So that you don't have to worry about cabling things together, boards strewn about your desk, etc. when prototyping and writing code.

#### DepthAI | Raspberry Pi HAT with FFC Camera Interfaces
![DepthAI Models](/images/67524974-078d1500-f66f-11e9-9b86-cd7578f63b42.png)
![DepthAI Models](/images/67516846-92b1df00-f65e-11e9-974b-b37825192901.png)
![DepthAI Models](/images/67516891-a8270900-f65e-11e9-9ad1-3318f49396e5.png)

This design allows you to leverage your existing Raspberry Pi (and perhaps even its mounting on your platform) and simply add this DepthAI hat to get this AI, depth, and vision processing power.  Off-board modular cameras allow integration up to 6 inches away from the Pi/HAT.

#### DepthAI | USB3 with FFC Camera Interfaces
![DepthAI Models](/images/67526422-f691d300-f671-11e9-8b11-e574e808c619.png)
![DepthAI Models](/images/67530493-745adc00-f67c-11e9-86bb-d78ba7150d16.png)
![DepthAI Models](/images/67530766-63f73100-f67d-11e9-9f9a-e7ca269832cb.png)

This design allows you to leverage DepthAI with whichever platform you choose.  It works with anything that runs OpenVINO, which is a lot of systems.

  - Linux: Ubuntu - Yocto Project - CentOS - Raspbian
  - Mac OS X
  - Windows 10
  
 Note: We have a preliminary system working which actually works with the Raspberry Pi Camera V2.1... but the ISP is still a bit broken so it flashes at lighting transitions/etc.  If there's sufficient interest we'll finish this proof-of-concept driver/camera tuning for the Myriad X and release hardware to support the Pi V2.1 camera as well.
 
 #### DepthAI | Modular (FFC) Cameras
 ![DepthAI Models](/images/67601447-a37f5500-f731-11e9-8c2c-dd7ca0ab9609.png)
 ![DepthAI Models](/images/67606323-41791c80-f73e-11e9-8b43-18e8d21e9070.png)
 
 We designed these modular cameras to have the same mounting pattern and board size as the Raspberry Pi V2.1 camera so that existing mounts/etc. likely can be used (or lightly modified and used).  That and, the size was going to be super close anyways, so why not go ahead and -not- make yet another mechanical mounting standard?
 
 ![DepthAI Models](/images/67602419-c448aa00-f733-11e9-905f-a288ea166a60.png)
 ![DepthAI Models](/images/67602612-34efc680-f734-11e9-9b74-adafa11a80fe.png)
 
 ## Use Cases

**Health and Safety**

The real-time and completely-on-device nature of DepthAI is what makes new use-cases in health and safety applications.  

Did you know that the number one cause of injury and death in oil and gas actually comes from situations of trucks and other vehicles impacting people? The size, weight, power, and real-time nature of DepthAI enables use-cases never before possible.

Imagine a smart helmet for factory workers that warns the worker when a fork-lift is about to run him or her over. Or even a smart fork-lift that can tell what objects are, where they are, and prevents the operator from running over the person - or hitting critical equipment. All while allowing the human operator to go about business as usual. The combination of Depth+AI allows such a system to real-time make 'virtual walls' around people, equipment, etc.

We're planning to use DepthAI internally to make **Commute Guardian**, which is a similar application, aiming to try to keep people who ride bikes safe from distracted drivers.

**Food processing**

DepthAI is hugely useful in food processing. To determine if a food product is safe, often many factors need to be taken into account, including size (volume), weight, and appearance. DepthAI allows some very interesting use-cases here. First, since it has real-time (at up to 120FPS) depth mapping, multiple DepthAI can be used to very accurately get the volume and weight of produce - without costly, error-prone mechanical weight sensors. And importantly, since mechanical weight sensors suffer from vibration error, etc., they limit how fast the food product can move down the line.

Using DepthAI for optical weighing and volume, the speed of the line can be increased significantly while also achieving a more accurate weight - with the supplemental data of full volumetric data - so you can sort with insane granularity.

And in addition, one of the most painful parts about inspecting food items with computer vision is that for many foods there's a huge variation of color, appearance, etc. that are all 'good' - so traditional algorithmic solutions fall apart (often resulting in 30% false-disposal rates when enabled, so they're disabled and teams of people do the inspection/removal by hand instead). But humans, looking at these food products can easily tell good/bad. And AI has been proven to be able to do the same.

So DepthAI would be able to weigh the food, get it's real-time size/shape - and be able to run a neural model real-time to produce good/bad criteria (and other grading) - which can be mechanically actuated to sort the food product real-time.

And most importantly, this is all quantified.  So not only can it achieve equivalent functionality of a team of people, it can also deliver data on size, shape, 'goodness', weight, etc. for every good product that goes through the line.

So you can have a record and quantify real-time and over time all the types of defects, the diseases seen, the packaging errors, etc. to be able to optimize all of the processes involved in the short-term, the long-term, and across seasonal variations.

**Manufacturing**

Similar to food processing, there are many places where DepthAI solves difficult problems that previously were not solvable with technology (i.e. required in-line human inspection and/or intervention) or where traditional computer vision systems do function, but are brittle, expensive, and require top experts in the field to develop and maintain the algorithms as products evolve and new products are added to the manufacturing line. 

DepthAI allows neural models to perform the same functions, while also measuring dimensions, size, shape, mass real-time - removing the need for personnel to do mind-numbing and error prone inspection while simultaneous providing real-time quantified business isights - and without the huge NRE required to pay for algorithmic solutions.  

**Mining**

This one is very interesting - as working in mines is very hazardous - but you often want/need human perception in the mine to know what to do next. DepthAI allows that sort of insight, without putting a human at risk.  So the state of the mine and of the mining equipment can be monitored real-time and quantified - giving alerts when things are going wrong (or right) - amplifying personnel's capability to keep people and equipment safe while increasing visibility into overall mining performance and efficiency.

**Autonomy**

When programming an autonomous platform to move about the world, the two key things needed are (1) what are the things around me and (2) what is their location relative to me. DepthAI gives this in an easy API which allows straightforward business logic for driving the platform.  

In the aerial use case, this includes drone sense-and-avoid, emergency recovery (where to land/crash without harming people or property if the prop fails and the UAV only has seconds to respond), and self-navigation in GPS-denied environments.

And for ground platforms, this allows unstructured navigation. So understanding what is around, and where, without a-priori knowledge, and responding accordingly.  

A good out-of-the-box example of this actually Unfolding Space ([here](https://hackaday.io/project/163784-unfolding-space)) (an early tester of DepthAI), which aims to aid in the autonomy of people who are blind.  With DepthAI, such a system no longer has to be simple 'avoid the nebulous blob over there' but rather, 'there's a park bench 2.5 meters to your left and all 5 seats are open'.

Another more straightforward example is autonomous lawn mowing, while safely avoiding unexpected obstacles.
 
 
 **Support Us!**
 Want to build things off this?  Or want to buy these straight up?  Support our CrowdSupply to make this a reality:
 https://www.crowdsupply.com/luxonis/depthai
 
 Best,
 
 Brandon and the Luxonis team!
 
 
 

 
 

