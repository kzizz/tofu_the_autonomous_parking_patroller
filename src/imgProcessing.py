#!/usr/bin/env python

from __future__ import print_function

import roslib
import numpy as np
import math
import sys
import rospy
from geometry_msgs.msg import Twist
import matplotlib.pyplot as plt
import cv2
import matplotlib.patches as patches
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from skimage.io import imread
from skimage.filters import threshold_otsu
from skimage import measure
from skimage.measure import regionprops

class img_processor:

    def __init__(self):
        #Publishing and subscribing
        self.image_pub = rospy.Publisher("image_topic_2",Image)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw",Image,self.callback)
        #CHANGE this to publishing to CNN:
        #self.velocity_cmd = rospy.Publisher('/R1/cmd_vel', Twist,queue_size=1)

    def callback(self,data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        rows,cols,channels = cv_image.shape

        IMAGE_H = rows
        IMAGE_W = cols
        #crop image to probable height of license plate
        warped_img = cv_image[rows-400:cols-400]
        
        #color masks and cropping     
        #blue for car detection 
        lowerBlue = np.array([0, 0, 0],dtype = "uint8") 
        upperBlue = np.array([255,30, 0],dtype = "uint8") 
        blueCarMask = cv2.inRange(warped_img, lowerBlue, upperBlue)
        blueCarMask = cv2.medianBlur(blueCarMask, 5)
        blueCarMask = cv2.erode(blueCarMask, None, iterations=2)
        blueCarOutput = cv2.bitwise_and(warped_img, warped_img, mask = blueCarMask)
        #Crop the image to the bounding box of the blue car
        blueCarBinary = self.make_binary_image(blueCarOutput)
        carCroppedImg=self.crop_image_only_outside_using_mask(blueCarBinary,warped_img,tol=0)
        # White for detecting license plates 
        lowerWhite = np.array([100, 100, 100],dtype = "uint8")
        upperWhite = np.array([200, 200, 200],dtype = "uint8")
        whiteMask = cv2.inRange(carCroppedImg, lowerWhite, upperWhite)
        whiteMask = cv2.medianBlur(whiteMask, 5)
        whiteMask = cv2.erode(whiteMask, None, iterations=2)
        #Crop the image to the bounding box of the stripe on the back of the car
        whiteOutput = cv2.bitwise_and(carCroppedImg, carCroppedImg, mask = whiteMask)
        whiteBinary = self.make_binary_image(whiteOutput)
        whiteCroppedImg=self.crop_image_only_outside_using_mask(whiteBinary,carCroppedImg,tol=0)
        whiteCroppedBinary = self.make_binary_image(whiteCroppedImg)
        whiteCroppedBinary = cv2.bitwise_not(whiteCroppedBinary)

        cv2.imshow("white cropped",whiteCroppedImg)
        cv2.waitKey(3)
        cv2.imshow("white output",whiteOutput)
        #invert white binary image so parking info and license are 1
        #Finding boundary boxes for P#, license, and QR code
        self.boundary_finder(whiteCroppedImg,whiteCroppedBinary)
        
        # #Use to detect plates with a mask - issue: QR code and P-# come up as blue aswell 
        # #blue for plate detection
        # lowerPlateBlue = np.array([0, 0, 0],dtype = "uint8") 
        # upperPlateBlue = np.array([255,50, 30],dtype = "uint8") #255,70,50
        # bluePlateMask = cv2.inRange(whiteCroppedImg, lowerPlateBlue, upperPlateBlue)
        # bluePlateOutput = cv2.bitwise_and(whiteCroppedImg, whiteCroppedImg, mask = bluePlateMask) 
        # cv2.imshow("plate mask",bluePlateOutput)
        # cv2.waitKey(3)  
        # bluePlateBinary = self.make_binary_image(bluePlateOutput)
        # cv2.imshow("plate binary",bluePlateBinary)
        # cv2.waitKey(3) 

        # bluePlateBinary = self.make_binary_image(bluePlateOutput)
        # plateCroppedImg=self.crop_image_only_outside_using_mask(bluePlateBinary,whiteCroppedImg,tol=0)
        # cv2.imshow("plate cropped",plateCroppedImg)
        # cv2.waitKey(3)

        #self.boundary_finder(grayCarImg,binaryCarImg)

    
    #find the boundaries of the license plate using connected component analysis
    def boundary_finder(self,img,binaryImg):
        #Find connected components
        labelImg = measure.label(binaryImg)

        #Find the expected dimmensions of a license plate
        plate_dimensions = (0.3*labelImg.shape[0], 0.6*labelImg.shape[0], 0.04*labelImg.shape[1], 0.12*labelImg.shape[1])
        min_height, max_height, min_width, max_width = plate_dimensions
        plate_objects_cordinates = []
        plate_like_objects = []

        # regionprops creates a list of properties of all the labelled regions
        for region in regionprops(labelImg):
            if region.area < 100:
            #if the region is so small then it's likely not a license plate
                continue

            ##FOLLOWING IS NON-RESTRICTIVE METHOD
            min_row, min_col, max_row, max_col = region.bbox
            rectBorder = cv2.rectangle(img, (min_col, min_row), (max_col, max_row), (255,0,0), 2)
            print("I found a rectangle!")

            # ##FOLLOWING IS MORE RESTRICTIVE METHOD
            # # the bounding box coordinates
            # min_row, min_col, max_row, max_col = region.bbox
            # region_height = max_row - min_row
            # region_width = max_col - min_col
            # # ensuring that the region identified satisfies the condition of a typical license plate
            # if region_height >= min_height and region_height <= max_height and region_width >= min_width and region_width <= max_width and region_width > region_height:
            #     plate_like_objects.append(binaryImg[min_row:max_row,min_col:max_col])
            #     plate_objects_cordinates.append((min_row, min_col,max_row, max_col))
            #     cv2.rectangle(img, (min_col, min_row), (max_col, max_row), (0,255,0), 2)
            #     print("I found a rectangle!")
        cv2.imshow("rectangles",img)
        cv2.waitKey(3) 
        cv2.imshow("binary",binaryImg)
        cv2.waitKey(3)

    def crop_image_only_outside_using_mask(self,des_mask,img,tol=0):
        # img is 2D image data
        # tol  is tolerance
        mask = des_mask>tol
        print (des_mask.shape)
        m,n = des_mask.shape
        mask0,mask1 = mask.any(0),mask.any(1)
        col_start,col_end = mask0.argmax(),n-mask0[::-1].argmax()
        row_start,row_end = mask1.argmax(),m-mask1[::-1].argmax()
        return img[row_start:row_end,col_start:col_end]

    def make_binary_image(self,img):
        #turn image into grayscale and binary
        grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        threshold_value = threshold_otsu(grayImg) #CHANGE this function throws an error if the mask is empty
        binaryImg = cv2.threshold(grayImg, threshold_value, 255, cv2.THRESH_BINARY)[1]
        return binaryImg


def main(args):
    rospy.init_node('img_processor', anonymous=True) #CHECK : Does this needed to be added to a world.launch file somewhere?
    imgP = img_processor()
    # try:
    rospy.spin()
    # except KeyboardInterrupt:
        # print("Shutting down")
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)












