import cv2
import numpy as np
import depthai      # Library necessary for communicating with DepthAI hardware.

def isolate_red_area(image):
    # Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) #convert image from BGR to HSV

    # Define the lower and upper bounds for red color in HSV
    # read in [Blue, Green, Red]
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255]) 

    # Create a binary mask using inRange() function
    mask = cv2.inRange(hsv_image, lower_red, upper_red) #create binary max based on lower and upper bounds and apply to hsv image
    # result = cv2.medianBlur(mask, 3) #apply median filter in case 
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)    # Apply the mask to the original image
    biggestFlame = 0
    index = 0
    for i in range(1, len(stats)): #find biggest connected object
        if stats[i][4] > biggestFlame:
            biggestFlame = stats[i][4]
            index = i
    x,y,w,h,_ = stats[index]  #(x,y) is top left corner, (x+w, y+h) is bottom right... cuz cartesian coordinate system
    centerx, centery = centroids[index] #extract center of object
    text_position = (int(centerx) - 30, int(centery) + 10)  # Adjust position for centering
    result = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 5)  # Draw rectangle around red object
    result = cv2.putText(result, "Fire", text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1) #add text at center of square
    binary = cv2.bitwise_and(result, image, mask=mask) #just for fun, bitwise and with original image to see just red object
# testgit
    return binary, result

def process_red_area_in_video():
    # Create a DepthAI pipeline. Pipeline tells DepthAI what operations to perform when running.  Define all of the resources used and flows here
    pipeline = depthai.Pipeline()

    # Want color camera as output: create a ColorCamera node in the pipeline
    cam_rgb = pipeline.createColorCamera()
    cam_rgb.setBoardSocket(depthai.CameraBoardSocket.RGB)
    cam_rgb.setResolution(depthai.ColorCameraProperties.SensorResolution.THE_1080_P)

    # XLinkOut is a "way out" from the device. 
    # Any data you want to transfer to host needs to be sent via XLink
    xout_rgb = pipeline.createXLinkOut() # Create an XLinkOut node for the camera preview
    xout_rgb.setStreamName("video") # Set XLink stream name to video. This is the rgb camera output.
    cam_rgb.preview.link(xout_rgb.input) # Linking camera preview to XLink input, so that the frames will be sent to host

    # Start the pipeline and search for an available device that will run the pipeline.
    with depthai.Device(pipeline) as device: 
        # device is now in "Running" mode, and will send data through XLink.

        video_queue = device.getOutputQueue(name="video", maxSize=1, blocking=False) # define an output queue that takes the XLink stream as input
        
        while True:
            # Get the latest frame from the Oak-D camera
            frame = video_queue.get()

            if frame is not None:
                # Convert the Oak-D camera frame from OpenCV format to BGR format
                bgr_frame = frame.getCvFrame()

                # Label the red area in the image
                justred, labeled_image = isolate_red_area(bgr_frame)

                # Show just red objects
                cv2.imshow("Isolated Red Object", justred)

                # Show original image with object framed and text
                cv2.imshow("Labeled Red Area", labeled_image)

            key = cv2.waitKey(1)
            if key == ord('q'):
                cv2.destroyAllWindows()
                break

if __name__ == '__main__':
    process_red_area_in_video()