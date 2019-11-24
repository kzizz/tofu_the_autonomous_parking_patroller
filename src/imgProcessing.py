#!/usr/bin/env python

from __future__ import print_function

import roslib
import numpy as np
import math
import sys
import rospy
from geometry_msgs.msg import Twist
import matplotlib.pyplot as plt
import time
import cv2
import datetime
import operator
import string
import keras
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from skimage.io import imread
from skimage.filters import threshold_otsu
from skimage import measure
from skimage.measure import regionprops
from numpy import loadtxt

from keras.models import load_model
from keras import backend
import tensorflow as tf


class img_processor:

    def __init__(self):
        #Publishing and subscribing
        self.image_pub = rospy.Publisher("image_topic_2",Image)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw",Image,self.callback)
        self.time = 0
        self.counter = 0
        config = tf.ConfigProto(
            device_count={'GPU': 1},
            intra_op_parallelism_threads=1,
            allow_soft_placement=True
            )
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.6
        self.session = tf.Session(config=config)
        keras.backend.set_session(self.session)
        
        self.model = load_model('/home/fizzer/enph353_ws/src/tofu_img_process/src/modelWithRealData.h5')
        # tf.reset_default_graph()
        self.model._make_predict_function()
        self.numberModel = load_model('/home/fizzer/enph353_ws/src/tofu_img_process/src/numberModel.h5')
        self.numberModel._make_predict_function()
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
        warped_img = cv_image[rows-300:rows,0:300]
        # Convert BGR to HSV
        hsv = cv2.cvtColor(warped_img, cv2.COLOR_BGR2HSV)
        # Threshold the HSV image to get only blue colors
        foundCar, blueCropped = self.lookForCar(hsv)
        if(foundCar is True):
            print("found car")
            self.findLicensePlate(blueCropped)

  

    def lookForCar(self,hsv):
        isCar = False
        lowerBlue = np.array([94,80,2])
        upperBlue = np.array([126,255,255])
        lowerWhite = np.array([0, 0, 90],dtype = "uint8")
        upperWhite = np.array([30, 50, 255],dtype = "uint8")
        # Threshold the HSV image to get only blue colors 
        blueCarMask = cv2.inRange(hsv, lowerBlue, upperBlue)
        whiteMask = cv2.inRange(hsv, lowerWhite, upperWhite)
        blueCarOutput = cv2.bitwise_and(hsv, hsv, mask = blueCarMask)
        whiteCarOutput = cv2.bitwise_and(hsv,hsv, mask = whiteMask)
        # cv2.imshow("cropped image",warped_img)
        cv2.imshow("blueCarOutput", blueCarOutput)
        cv2.waitKey(3)
        cv2.imshow("whiteCarOutput",whiteCarOutput)
        cv2.waitKey(3)
        bluePercentage =np.divide(float(np.count_nonzero(blueCarOutput)),float(np.count_nonzero(hsv)))
        whitePercentage =np.divide(float(np.count_nonzero(whiteCarOutput)),float(np.count_nonzero(hsv)))
        croppedBlueImage = 0
        # if(self.counter % 5 == 0):
        #     print("blue percentage")
        #     print(bluePercentage)
        #     print("white percentage")
        #     print(whitePercentage)
        # if(bluePercentage > 0.09 and bluePercentage < 0.135):
        if(bluePercentage > 0.09 and bluePercentage < 0.5 and whitePercentage > 0.1):
            print("blue percentage of the car")
            print(bluePercentage)
            print("found a car")
            cv2.imshow("car",blueCarOutput)
            isCar = True
            blueCarBinary = self.make_binary_image(blueCarOutput)
            croppedBlueImage=self.crop_image_only_outside_using_mask(blueCarBinary,blueCarOutput,tol=0)
            cv2.imshow("carCroppedImgBlue", croppedBlueImage)
        self.counter += 1 
        return isCar,croppedBlueImage

    # returns images to feed to the neural network in the right order
    def findLicensePlate(self,blueImage): 
            croppedBlueCarBinary = self.make_binary_image(blueImage)
            regions = self.boundary_finder(blueImage,croppedBlueCarBinary)
            #Define an empty dictionary to associate image with the order it apears on license
            numberImages = {}
            validRegions = self.getValidRegions(regions)
            orderedImages = []
            if(len(validRegions) == 0):
                print("no license plates found")
            else:
                for validRegion in validRegions:
                    min_row = validRegion[0]
                    max_row = validRegion[1]
                    min_col = validRegion[2]
                    max_col = validRegion[3]
                    # print("minimum column")
                    # print(min_col)
                    cropped_img = blueImage[min_row-3:max_row+3,min_col-3:max_col+3].copy()
                    numberImages[(min_col + max_col) / 2]= (cv2.cvtColor(cropped_img,cv2.COLOR_HSV2BGR))
                
                #sort the images based on their min_col
                sortedNumberImages = sorted(numberImages.keys())
                # print("Number Images")
                # print(sortedNumberImages)

                # print("%s: %s"% (sortedNumberImages[0],numberImages[sortedNumberImages[0]] ))
                for i in range(2):
                    # print(sortedNumberImages[i])
                    value = self.readLetter(numberImages[sortedNumberImages[i]])
                    print(value)
                for i in range (2,4):
                    value = self.readNumber(numberImages[sortedNumberImages[i]])
                    print(value)   
                timestamp = str(datetime.datetime.now().strftime("%Y%m%d_%H-%M-%S"))
                cv2.imshow("first number",numberImages[sortedNumberImages[0]])
                # cv2.imwrite("licenseLetters/"+timestamp+"_1.png",numberImages[sortedNumberImages[0]])
                cv2.imshow("second number",numberImages[sortedNumberImages[1]])
                # cv2.imwrite("licenseLetters/"+timestamp+"_2.png",numberImages[sortedNumberImages[1]])
                cv2.imshow("third number",numberImages[sortedNumberImages[2]])
                # cv2.imwrite("licenseLetters/"+timestamp+"_3.png",numberImages[sortedNumberImages[2]])
                cv2.imshow("fourth number",numberImages[sortedNumberImages[3]])
                # cv2.imwrite("licenseLetters/"+timestamp+"_4.png",numberImages[sortedNumberImages[3]])
    # checks the validity of the regions that could be numbers and letters on the license plate
    #returns regions in the format of min_row, max_row, min_col, max_col
    def getValidRegions (self,regions):
        validRegions = []
        if(len(regions) == 0):
                print("no license plates found")
        elif(len(regions) is 2 or len(regions) is 3):
            print("length of region is 2 or 3")
            for region in regions:
                min_row, min_col, max_row, max_col = region.bbox
                width = abs(max_col - min_col)
                height = abs(max_row - min_row)
                aspectRatio = float(width) / float(height)
                print(aspectRatio)
                if aspectRatio > 2:
                    # two letters or numbers in one line, we should split it
                    region1 = [min_row , max_row, min_col, min_col + width / 2 -1 ]
                    region2 = [min_row, max_row, min_col + width / 2 + 1, max_col]
                    validRegions.extend([region1,region2])
                else:
                    reformatRegion = [min_row, max_row, min_col, max_col]
                    validRegions.append(reformatRegion)
            if(len(validRegions) !=4):
                validRegions = [] #not a license plate no valid region
                print("not a valid region")
                
        elif(len(regions) is 4):
                for region in regions:
                    min_row, min_col, max_row, max_col = region.bbox
                    validRegion = [min_row,max_row, min_col, max_col]
                    validRegions.append(validRegion)
        return validRegions
    def readLetter(self,img):
        labels = ['0','1','2','3','4','5','6','7','8','9']
        labels.extend(list(string.ascii_uppercase))
        dictionary = {"image" : [] , "vector": [], "label": []}
        # print(img.shape)
        grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img_aug = np.repeat(grayImg[..., np.newaxis], 3, -1)
        try:
            img_aug = cv2.resize(img_aug,(18,22))
        except(e):
            print(e)
        cv2.imshow("aug img",img_aug)
        img_aug = np.expand_dims(img_aug, axis=0)
        print(img_aug.shape)
        with self.session.as_default():
            with self.session.graph.as_default():
                y_predict = self.model.predict(img_aug)[0]
                predictVal = max(y_predict)
                predictedVal_index = np.where(y_predict == predictVal)[0][0]
                predictedVal = labels[predictedVal_index]
        return predictedVal
    def readNumber(self,img):
        labels = ['0','1','2','3','4','5','6','7','8','9']
        dictionary = {"image" : [] , "vector": [], "label": []}
        # print(img.shape)
        grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img_aug = np.repeat(grayImg[..., np.newaxis], 3, -1)
        try:
            img_aug = cv2.resize(img_aug,(18,22))
        except(e):
            print(e)
        cv2.imshow("aug img",img_aug)
        img_aug = np.expand_dims(img_aug, axis=0)
        print(img_aug.shape)
        with self.session.as_default():
            with self.session.graph.as_default():
                y_predict = self.numberModel.predict(img_aug)[0]
                predictVal = max(y_predict)
                predictedVal_index = np.where(y_predict == predictVal)[0][0]
                predictedVal = labels[predictedVal_index]
        return predictedVal
    #find the boundaries of the license plate using connected component analysis
    def boundary_finder(self,img,binaryImg):
        #Find connected components
        labelImg = measure.label(binaryImg)
        recImg = img.copy()
        plate_objects_cordinates = []
        plate_like_objects = []
        regions = []
        # regionprops creates a list of properties of all the labelled regions
        for region in regionprops(labelImg):
            if region.area < 90 or region.area > 400:
            #if the region is so small then it's likely not a license plate
                continue

            regions.append(region)
            min_row, min_col, max_row, max_col = region.bbox
            rectBorder = cv2.rectangle(recImg, (min_col, min_row), (max_col, max_row), (0,255,0), 2)
            # print("I found a rectangle!")
        
        cv2.imshow("rectangles",recImg)
        cv2.waitKey(3) 
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
        threshold_value = threshold_otsu(grayImg)
        binaryImg = cv2.threshold(grayImg, threshold_value, 255, cv2.THRESH_BINARY)[1]

        return binaryImg


def main(args):
    imgP = img_processor()
    # try:
    rospy.spin()
    # except KeyboardInterrupt:
        # print("Shutting down")
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    rospy.init_node('img_processor', anonymous=True) #CHECK : Does this needed to be added to a world.launch file somewhere?
    main(sys.argv)












