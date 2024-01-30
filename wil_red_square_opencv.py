import cv2
import numpy as np

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

    return binary, result

def process_red_area_in_video():
    cap = cv2.VideoCapture(0) # capture video with laptop camera
    while True:
        _, frame = cap.read() #get frame
        # cv2.imshow("Original Image", frame) #show original frame

    # Label the red area in the image
        justred, labeled_image = isolate_red_area(frame) 
        cv2.imshow("Isolated Red Object", justred) #show just red objects
        cv2.imshow("Labeled Red Area", labeled_image) #show original image with object framed and text
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            cv2.destroyAllWindows()
            break
        
        
if __name__ == '__main__':
    process_red_area_in_video()