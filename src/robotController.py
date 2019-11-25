#!/usr/bin/env python

from __future__ import print_function
import time
import roslib
import numpy as np
import math
#roslib.load_manifest('my_package')
import sys
import rospy
from geometry_msgs.msg import Twist
import matplotlib.pyplot as plt
import cv2
import time
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from skimage.measure import compare_ssim


class robot_controller:
    #This will be our state machine

    def __init__(self):
        self.image_pub = rospy.Publisher("image_topic_2",Image)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw",Image,self.callback)
        self.velocity_cmd = rospy.Publisher('/R1/cmd_vel', Twist,queue_size=1)
        self.targetOffset = 450
        self.sawCrosswalk = False
        self.atCrosswalk = False
        self.offCrosswalk = True
        self.state =  "initializing" #CHANGE "initializing"
        self.GO_STRAIGHT = self.targetOffset
        self.TURN_LEFT = 0
        self.initDoneStraight = False
        self.pedCounter = 0
        self.prevPedView = 0
        self.prevPedScore = 100
        self.scores = []
        self.pedTimer = 0

    def callback(self,data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        
        rows,cols,channels = cv_image.shape
        IMAGE_H = rows
        IMAGE_W = cols
        warped_img = cv_image[rows-200:, cols-400:cols] 
        crosswalkImage = cv_image[rows-200:,0:cols]
        pedImage = cv_image[rows-500:,0:cols]
        
        #color masks 
        #detecting lines on the street
        lowerWhite = np.array([250, 250, 250],dtype = "uint8")
        upperWhite = np.array([255, 255, 255],dtype = "uint8")
        whiteMask = cv2.inRange(warped_img, lowerWhite, upperWhite)
        whiteMask = cv2.medianBlur(whiteMask, 5)
        whiteMask = cv2.erode(whiteMask, None, iterations=2)

        #detecting the street
        lowerGray = np.array([50, 80, 50],dtype = "uint8")
        upperGray = np.array([190, 90, 90],dtype = "uint8")
        grayMask = cv2.inRange(crosswalkImage, lowerGray, upperGray)
        #grass green
        lowerGreen = np.array([10,70,10],dtype = "uint8")
        upperGreen = np.array([70,210,30],dtype = "uint8")
        greenMask = cv2.inRange(warped_img, lowerGreen, upperGreen)
        #red for cross walk
        lowerRed = np.array([0, 0, 255-20],dtype = "uint8")
        upperRed = np.array([255, 20, 255],dtype = "uint8")
        redMask = cv2.inRange(crosswalkImage, lowerRed, upperRed)
        #blue for car detection
        lowerBlue = np.array([0, 0, 0],dtype = "uint8")
        upperBlue = np.array([255, 30, 20],dtype = "uint8")
        blueMask = cv2.inRange(warped_img, lowerBlue, upperBlue)
        #apply masks 
        greenOutput = cv2.bitwise_and(warped_img, warped_img, mask = greenMask)
        redOutput = cv2.bitwise_and(crosswalkImage, crosswalkImage, mask = redMask)
        grayOutput = cv2.bitwise_and(crosswalkImage, crosswalkImage, mask = grayMask)
        whiteOutput = cv2.bitwise_and(warped_img, warped_img, mask = whiteMask)
        blueOutput = cv2.bitwise_and(warped_img, warped_img, mask = blueMask)
        
        #Find cropped images for driving
        grayWarped = cv2.cvtColor(whiteOutput,cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(grayWarped, 20, 255, 0)
        img, contours, hierarchy = cv2.findContours(thresh, 1, 2)
            
        #check if we can see the red line indicating a cross walk
        redPixelCount = np.count_nonzero(redOutput)
        print(redPixelCount)
        cv2.imshow("redOutput",redOutput)
        cv2.waitKey(3)
        #Find center of mass for initialization & driving
        M = cv2.moments(img)
        middlePixel = 0
        cX = 0
        cY = 0
        if(M["m00"] == 0):
            offset = self.targetOffset
            #Can't make equal to the target offset here, otherwise init won't work
            if(self.state == "initializing"):
                print ("m00=0, offset 0")
                offset = 0 #CHECK IF THIS WORKS
            else:
                print ("m00=0, offset t.o")
                offset = self.targetOffset 

        else:
            cX = cols - 400 + int(M["m10"]/M["m00"])
            cY = rows - 200 + int(M["m01"]/ M["m00"])
            middlePixel = cols/2
            offset = cX - middlePixel
        #Draw circles on image for troubleshooting
        cv2.circle(cv_image,(middlePixel,cY), 5, (255,0,0))
        cv2.circle(cv_image, (cX,cY), 5, (255,0,0))
        # cv2.imshow("Image window", redOutput)
        # cv2.waitKey(3)
        # cv2.imshow("contour image",cv_image)
        # cv2.waitKey(3)
        # cv2.imshow("crosswalk view",crosswalkImage)
        # cv2.imshow("whiteOutput", whiteOutput)

        #State machine for driving
        if(self.state == "initializing"):
            middlePixel = cols / 2
            initImage = cv_image[rows-300:,middlePixel-100:middlePixel+100]
            initWhiteMask = cv2.inRange(initImage,lowerWhite,upperWhite)
            initWhiteOutput = cv2.bitwise_and(initImage, initImage, mask = initWhiteMask)
            initWhitePercentage = np.divide(float(np.count_nonzero(initWhiteOutput)) , float(np.count_nonzero(initImage)))            
            initComplete = False
            print("Initializing")
            print("White Percentage:")
            print(initWhitePercentage)
            # cv2.imshow("init white output",initWhiteOutput)
            # cv2.waitKey(3)
            #DONT CHANGE THIS VAL: If white percentage is less than 10%, we haven't gone straight long enough and should keep going
            if (initWhitePercentage < 0.03 and self.initDoneStraight == False):
                print("Going straight to init")
                self.pid(self.GO_STRAIGHT)

            else:
                print("Offset")
                print(offset)
                self.initDoneStraight = True
                print("else")
                #If still facing the line head on, turn left.  Done to compensate for cropping only the right side of the image for line following.
                if(initWhitePercentage > 0.015):
                    print("Still facing - Turning to init")
                    self.pid(self.TURN_LEFT)
                #If you've lined up with right lane line, drive on
                elif((abs(offset)<self.targetOffset+60) and (abs(offset)>self.targetOffset-60)):
                    print("Done init!")
                    initComplete = True
                #Keep turning until tofu is in line with right lane line
                else:
                    print("Turning to init")
                    self.pid(self.TURN_LEFT)

            if (initComplete == True):
                self.state = "driving"
            else:
                self.state = "initializing"

        if (self.state == "driving"): 
            print("Driving...")
            self.pid(offset) 
            if(redPixelCount < 20000):
                    self.state = "driving"
            else:
                    self.state = "entering_crosswalk"
            
        
        elif (self.state == "entering_crosswalk"):
            print("Entering crosswalk...")
            self.pid(offset)
            if(redPixelCount>20000):
                self.state = "entering_crosswalk"
            else:
                self.state = "waiting_for_ped"
                pedImage = cv2.cvtColor(cv_image[rows-300:rows-200,350:cols-350],cv2.COLOR_BGR2GRAY)
                self.prevPedView = pedImage
                self.pedTimer = time.time()

        elif (self.state == "waiting_for_ped"):
            self.stop()
            pedImage = cv2.cvtColor(cv_image[rows-300:rows-200,350:cols-350],cv2.COLOR_BGR2GRAY)
            if (self.pedCounter % 3 == 0):
                (score,diff) = compare_ssim(pedImage,self.prevPedView, full=True)
                # cv2.imshow("prevPedView",self.prevPedView)
                # print("difference in score")
                # print(abs(score - self.prevPedScore))
                diffScore = abs(score - self.prevPedScore) 
                self.scores.append(diffScore)
                if(time.time() - self.pedTimer > 4.5):
                    lastThreeDiff = self.scores[len(self.scores)-6:]
                    averageDiff = sum(lastThreeDiff) / len(lastThreeDiff)
                    print("average diff")
                    print(averageDiff)
                    if(averageDiff < 0.0005):
                        self.state = "on_crosswalk"
                self.prevPedScore = score
                self.prevPedView = pedImage
            # pedWhiteMask = cv2.inRange(pedImage, lowerWhite, upperWhite)
            # pedWhitePercentage = np.divide(float(np.count_nonzero(pedWhiteMask)) , float(np.count_nonzero(pedImage)))            
            # print("Waiting for pedestrian...")

            # cv2.imshow("PedView",pedImage)
            # cv2.waitKey(3)

        elif (self.state == "on_crosswalk"):
            self.pedTimer = time.time() #resetting the timer for pedestrian
            #Drive based on grey mask so that crosswalk white lines don't cause erratic driving
            grayWarpedCross = cv2.cvtColor(grayOutput,cv2.COLOR_BGR2GRAY)
            ret,thresh = cv2.threshold(grayWarpedCross, 20, 255, 0)
            img, contours, hierarchy = cv2.findContours(thresh, 1, 2)
            M = cv2.moments(img)
            if(M["m00"] == 0):
                offset = self.targetOffset
            else:
                cX = int(M["m10"]/M["m00"])
                cY = int(M["m01"]/ M["m00"])
                middlePixel = cols/2
                offset = cX - middlePixel + self.targetOffset
                print("Offset")
                print(offset)
            self.pid(offset)
            print("On crosswalk...")
            if (redPixelCount <20000):
                self.state = "on_crosswalk"
            else:
                self.state = "exiting_crosswalk"
        elif (self.state == "exiting_crosswalk"):
            self.pid(offset)
            print("Exiting crosswalk...")
            if (redPixelCount > 20000):
                self.state = "exiting_crosswalk"
            else:
                self.state = "driving" 
        self.pedCounter += 1   
            

    def pid(self,offset):
        differenceTolerance = 55
        angularScale = 5
        xVelocity = 0.03
        zTwist = 0.0
        offsetOvershoot = self.targetOffset - offset
        if(abs(offsetOvershoot) > differenceTolerance):
            xVelocity = 0.0
            zTwist = angularScale * offsetOvershoot
        vel_msg = Twist()
        vel_msg.linear.x = xVelocity
        vel_msg.angular.z = zTwist
        self.velocity_cmd.publish(vel_msg)       
    
    def stop(self):
        xVelocity = 0.00
        zTwist = 0.0
        vel_msg = Twist()
        vel_msg.linear.x = xVelocity
        vel_msg.angular.z = zTwist
        self.velocity_cmd.publish(vel_msg)   


def main(args):
    rc = robot_controller()
    rospy.spin()

if __name__ == '__main__':
    rospy.init_node('robot_controller', anonymous=True)
    main(sys.argv)












