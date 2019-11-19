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
import datetime
import operator
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
        # warped_img = cv_image[rows-400:cols-400]
        warped_img = cv_image[rows-300:rows,0:300]
        
        #color masks and cropping     
        #blue for car detection 
        # Convert BGR to HSV
        hsv = cv2.cvtColor(warped_img, cv2.COLOR_BGR2HSV)

        # define range of blue color in HSV
        # lower_blue = np.array([110,50,50])
        # upper_blue = np.array([130,255,255]) 
        lowerBlue = np.array([94,80,2])
        upperBlue = np.array([126,255,255])
        # Threshold the HSV image to get only blue colors
        # mask = cv2.inRange(hsv, lower_green, upper_green)
        # lowerBlue = np.array([0, 0, 0],dtype = "uint8") 
        # upperBlue = np.array([255,30, 0],dtype = "uint8") #can pick up Parking values
        blueCarMask = cv2.inRange(hsv, lowerBlue, upperBlue)
        # blueCarMask = cv2.medianBlur(blueCarMask, 5)
        # blueCarMask = cv2.erode(blueCarMask, None, iterations=2)
        blueCarOutput = cv2.bitwise_and(hsv, hsv, mask = blueCarMask)
        cv2.imshow("cropped image",warped_img)
        cv2.imshow("blueCarOutput", blueCarOutput)
        cv2.waitKey(3)
        bluePercentage =float( np.count_nonzero(np.asarray(blueCarOutput))) / float(np.count_nonzero(np.asarray(warped_img)))
        # print("blue percentage")
        # print(bluePercentage)
        # if(bluePercentage > 0.09 and bluePercentage < 0.135):
        if(bluePercentage > 0.1 and bluePercentage < 0.2):
            print("blue percentage of the car")
            print(bluePercentage)
            print("found a car")
            cv2.imshow("car",blueCarOutput)
            blueCarBinary = self.make_binary_image(blueCarOutput)
            carCroppedImgBlue=self.crop_image_only_outside_using_mask(blueCarBinary,blueCarOutput,tol=0)
            self.findLicensePlate(carCroppedImgBlue)
            
        # if (bluePercentage > 0.02):            
        #     # print(blueCarOutput.size)
        #     # print("in the blue loop")
        #     #print(len(np.nonzero(blueCarOutput)[0]))
        #     #Crop the image to the bounding box of the blue car
        #     blueCarBinary = self.make_binary_image(blueCarOutput)
        #     # cv2.imshow("blueCarOutput", blueCarOutput)
        #     carCroppedImg=self.crop_image_only_outside_using_mask(blueCarBinary,warped_img,tol=0)
        #     # White for detecting license plates 
        #     lowerWhite = np.array([100, 100, 100],dtype = "uint8")
        #     upperWhite = np.array([200, 200, 200],dtype = "uint8")
        #     whiteMask = cv2.inRange(carCroppedImg, lowerWhite, upperWhite)
        #     whiteMask = cv2.medianBlur(whiteMask, 5)
        #     whiteMask = cv2.erode(whiteMask, None, iterations=2)
        #     #Crop the image to the bounding box of the stripe on the back of the car
        #     whiteOutput = cv2.bitwise_and(carCroppedImg, carCroppedImg, mask = whiteMask)
        #     if (len(np.nonzero(whiteOutput)[0])):          
        #         lowerWhite = np.array([100, 100, 100],dtype = "uint8")
        #         upperWhite = np.array([200, 200, 200],dtype = "uint8")
        #         whiteMask = cv2.inRange(carCroppedImg, lowerWhite, upperWhite)
        #         whiteMask = cv2.medianBlur(whiteMask, 5)
        #         whiteMask = cv2.erode(whiteMask, None, iterations=2)
        #         #Crop the image to the bounding box of the stripe on the back of the car
        #         whiteOutput = cv2.bitwise_and(carCroppedImg, carCroppedImg, mask = whiteMask)
        #         whiteBinary = self.make_binary_image(whiteOutput)
        #         whiteCroppedImg=self.crop_image_only_outside_using_mask(whiteBinary,carCroppedImg,tol=0)
        #         whiteCroppedBinary = self.make_binary_image(whiteCroppedImg)
        #         whiteCroppedBinary = cv2.bitwise_not(whiteCroppedBinary)

        #         cv2.imshow("white cropped",whiteCroppedImg)
        #         cv2.waitKey(3)
        #         cv2.imshow("white output",whiteOutput)
        #         #invert white binary image so parking info and license are 1
        #         #Finding boundary boxes for P#, license, and QR code
        #         self.boundary_finder(whiteCroppedImg,whiteCroppedBinary)

    def findLicensePlate(self,blueImage): 
            croppedBlueCarBinary = self.make_binary_image(blueImage)
            regions = self.boundary_finder(blueImage,croppedBlueCarBinary)
            #Define an empty dictionary to associate image with the order it apears on license
            numberImages = {}

            if(len(regions) == 0):
                print("no license plates found")
            elif(len(regions) != 4):
                print("could not catch all the numbers or this is not a license plate")
            else:
                for region in regions:
                    min_row, min_col, max_row, max_col = region.bbox
                    print("minimum column")
                    print(min_col)
                    cropped_img = blueImage[min_row-3:max_row+3,min_col-3:max_col+3].copy()
                    numberImages[min_col]= (cv2.cvtColor(cropped_img,cv2.COLOR_HSV2BGR))
                
                #sort the images based on their min_col
                sortedNumberImages = sorted(numberImages.keys())
                print("Number Images")
                print(sortedNumberImages)
                print("%s: %s"% (sortedNumberImages[0],numberImages[sortedNumberImages[0]] ))

                timestamp = str(datetime.datetime.now().strftime("%Y%m%d_%H-%M-%S"))
                cv2.imshow("first number",numberImages[sortedNumberImages[0]])
                cv2.imwrite("licenseLetters/"+timestamp+"_1.png",numberImages[sortedNumberImages[0]])
                cv2.imshow("second number",numberImages[sortedNumberImages[1]])
                cv2.imwrite("licenseLetters/"+timestamp+"_2.png",numberImages[sortedNumberImages[1]])
                cv2.imshow("third number",numberImages[sortedNumberImages[2]])
                cv2.imwrite("licenseLetters/"+timestamp+"_3.png",numberImages[sortedNumberImages[2]])
                cv2.imshow("fourth number",numberImages[sortedNumberImages[3]])
                cv2.imwrite("licenseLetters/"+timestamp+"_4.png",numberImages[sortedNumberImages[3]])


        
    #find the boundaries of the license plate using connected component analysis
    def boundary_finder(self,img,binaryImg):
        #Find connected components
        labelImg = measure.label(binaryImg)
        recImg = img.copy()
        #Find the expected dimmensions of a license plate
        # plate_dimensions = (0.3*labelImg.shape[0], 0.6*labelImg.shape[0], 0.04*labelImg.shape[1], 0.12*labelImg.shape[1])
        # min_height, max_height, min_width, max_width = plate_dimensions
        plate_objects_cordinates = []
        plate_like_objects = []
        regions = []
        # regionprops creates a list of properties of all the labelled regions
        for region in regionprops(labelImg):
            if region.area < 50 or region.area > 400:
            #if the region is so small then it's likely not a license plate
                continue

            ##FOLLOWING IS NON-RESTRICTIVE METHOD
            regions.append(region)
            min_row, min_col, max_row, max_col = region.bbox
            rectBorder = cv2.rectangle(recImg, (min_col, min_row), (max_col, max_row), (0,255,0), 2)
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
        cv2.imshow("rectangles",recImg)
        cv2.waitKey(3) 
        # cv2.imshow("binary",binaryImg)
        # cv2.waitKey(3)

        return regions


    def crop_image_only_outside_using_mask(self,des_mask,img,tol=0):
        # img is 2D image data
        # tol  is tolerance
        mask = des_mask>tol
        m,n = des_mask.shape
        mask0,mask1 = mask.any(0),mask.any(1)
        col_start,col_end = mask0.argmax(),n-mask0[::-1].argmax()
        row_start,row_end = mask1.argmax(),m-mask1[::-1].argmax()
        return img[row_start:row_end,col_start:col_end]

    def make_binary_image(self,img):
        #turn image into grayscale and binary
        grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        threshold_value = threshold_otsu(grayImg) #CHANGE this function throws an error if the greyImg is empty
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












