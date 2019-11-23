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



# load model


class img_processor:

    def __init__(self):
        #Publishing and subscribing
        self.image_pub = rospy.Publisher("image_topic_2",Image)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw",Image,self.callback)
        self.time = 0
        # self.model_sub = rospy.Subscriber("readLetters", String, callback)
        # config = tf.ConfigProto(
        #     device_count={'GPU': 1},
        #     intra_op_parallelism_threads=1,
        #     allow_soft_placement=True
        #     )
        # config.gpu_options.allow_growth = True
        # config.gpu_options.per_process_gpu_memory_fraction = 0.6
        # self.session = tf.Session(config=config)
        # keras.backend.set_session(self.session)
        
        # self.model = load_model('/home/fizzer/enph353_ws/src/tofu_img_process/src/modelWithRealData.h5')
        # # tf.reset_default_graph()
        # self.model._make_predict_function()
        #CHANGE this to publishing to CNN:
        #self.velocity_cmd = rospy.Publisher('/R1/cmd_vel', Twist,queue_size=1)
        self.counter = 0
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
        #crop image to probably height of parking number
        parking_img = cv_image[rows-300:rows-225,0:300]
        cv2.imshow("View", warped_img) #CHANGE
        cv2.imshow("Parking View", parking_img)
        cv2.waitKey(3)
        # Convert BGR to HSV
        hsv = cv2.cvtColor(warped_img, cv2.COLOR_BGR2HSV)
        # Threshold the HSV image to get only blue colors
        foundCar = self.lookForCar(hsv)
        if(foundCar is True):
            print("found car")
            parkingNum = self.find_parking_number(parking_img)
  

    def lookForCar(self,hsv):
        isCar = False
        lowerBlue = np.array([94,80,2])
        upperBlue = np.array([126,255,255])
        lowerWhite = np.array([0, 0, 90],dtype = "uint8")
        upperWhite = np.array([30, 50, 255],dtype = "uint8")
        # Threshold the HSV image to get only blue colors
        # mask = cv2.inRange(hsv, lower_green, upper_green)
        # lowerBlue = np.array([0, 0, 0],dtype = "uint8") 
        # upperBlue = np.array([255,30, 0],dtype = "uint8") #can pick up Parking values
        blueCarMask = cv2.inRange(hsv, lowerBlue, upperBlue)
        whiteMask = cv2.inRange(hsv, lowerWhite, upperWhite)
        blueCarOutput = cv2.bitwise_and(hsv, hsv, mask = blueCarMask)
        whiteCarOutput = cv2.bitwise_and(hsv,hsv, mask = whiteMask)
        # cv2.imshow("cropped image",warped_img)
        # cv2.imshow("blueCarOutput", blueCarOutput)
        # cv2.waitKey(3)
        # cv2.imshow("whiteCarOutput",whiteCarOutput)
        # cv2.waitKey(3)
        bluePercentage =np.divide(float(np.count_nonzero(blueCarOutput)),float(np.count_nonzero(hsv)))
        whitePercentage =np.divide(float(np.count_nonzero(whiteCarOutput)),float(np.count_nonzero(hsv)))
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
            # cv2.imshow("car",blueCarOutput)
            isCar = True
            # blueCarBinary = self.make_binary_image(blueCarOutput)
            # carCroppedImgBlue=self.crop_image_only_outside_using_mask(blueCarBinary,blueCarOutput,tol=0)
            # self.findLicensePlate(carCroppedImgBlue)
        self.counter += 1 
        return isCar

    def findLicensePlate(self,blueImage): 
            croppedBlueCarBinary = self.make_binary_image(blueImage)
            regions = self.boundary_finder(blueImage,croppedBlueCarBinary)
            #Define an empty dictionary to associate image with the order it apears on license
            numberImages = {}

            if(len(regions) == 0):
                print("no license plates found")
            # elif(len(regions) != 4):
            #     print("could not catch all the numbers or this is not a license plate")
            else:
                for region in regions:
                    min_row, min_col, max_row, max_col = region.bbox
                    # print("minimum column")
                    # print(min_col)
                    cropped_img = blueImage[min_row-3:max_row+3,min_col-3:max_col+3].copy()
                    numberImages[min_col]= (cv2.cvtColor(cropped_img,cv2.COLOR_HSV2BGR))
                
                #sort the images based on their min_col
                sortedNumberImages = sorted(numberImages.keys())
                print("Number Images")
                print(sortedNumberImages)
                # print("%s: %s"% (sortedNumberImages[0],numberImages[sortedNumberImages[0]] ))
                # for i in range(len(sortedNumberImages)):
                #     value = self.readLetter(numberImages[sortedNumberImages[i]])
                #     print(value)


                timestamp = str(datetime.datetime.now().strftime("%Y%m%d_%H-%M-%S"))
                # cv2.imshow("first number",numberImages[sortedNumberImages[0]])
                # cv2.imwrite("licenseLetters/"+timestamp+"_1.png",numberImages[sortedNumberImages[0]])
                # cv2.imshow("second number",numberImages[sortedNumberImages[1]])
                # cv2.imwrite("licenseLetters/"+timestamp+"_2.png",numberImages[sortedNumberImages[1]])
                # cv2.imshow("third number",numberImages[sortedNumberImages[2]])
                # cv2.imwrite("licenseLetters/"+timestamp+"_3.png",numberImages[sortedNumberImages[2]])
                # cv2.imshow("fourth number",numberImages[sortedNumberImages[3]])
                # cv2.imwrite("licenseLetters/"+timestamp+"_4.png",numberImages[sortedNumberImages[3]])

    #Inputs:self, and a cropped image of the of Tofu's view where a parking num could be located (left side)
    #Outputs: returns a cropped image of the parking number, to be submitted to the CNN
    def find_parking_number(self, img):
        #Define mask
        # lower = np.array([0,0,0])
        # upper = np.array([0,0,95])
        # numMask = cv2.inRange(img, lower, upper)
        # maskOutput = cv2.bitwise_and(img,img, mask = numMask)
        # maskOutput = cv2.cvtColor(maskOutput, cv2.COLOR_HSV2BGR)
        # #maskOutput = cv2.bitwise_not(maskOutput)
        # cv2.imshow("ParkingNumView",maskOutput)
        # #Make image binary 
        # binary = self.make_binary_image(maskOutput)
        # cv2.imshow("Binary Parking", binary)
        maskOutput = cv2.bitwise_not(self.make_binary_image(img))
        binary = maskOutput
        #find bounding boxes on parking numbers
        regions = self.boundary_finder(maskOutput,binary)
        #Define an empty dictionary to associate image with the order it apears on license
        numberImages = {}
        cv2.imshow("Binary Parking", maskOutput)
        

        if(len(regions) == 0):
            print("no parking num found")
        elif(len(regions) != 2):
            print("could not catch all the numbers or this is not a parking number")
        else:
            for region in regions:
                min_row, min_col, max_row, max_col = region.bbox
                #Filter the regions based on size
                width = max_col-min_col
                height = max_row-min_row
                print("width:")
                print(max_col-min_col)
                print("height:")
                print(max_row-min_row)
                cropped_img = img[min_row-3:max_row+3,min_col-3:max_col+3].copy()
                numberImages[min_col]= cropped_img

            sortedNumberImages = sorted(numberImages.keys())
            print("Found the right number of regions!!")
            timestamp = str(datetime.datetime.now().strftime("%Y%m%d_%H-%M-%S"))
            cv2.imshow("second char",numberImages[sortedNumberImages[1]])

        print("Number of regions")
        print(len(numberImages))

        
        #     # cv2.imshow("second char",numberImages[sortedNumberImages[1]])
        #     # cv2.imwrite("licenseLetters/"+timestamp+"_1.png",numberImages[sortedNumberImages[0]])
        #     # cv2.imshow("second number",numberImages[sortedNumberImages[1]])
        #     # cv2.imwrite("licenseLetters/"+timestamp+"_2.png",numberImages[sortedNumberImages[1]])
        #     # cv2.imshow("third number",numberImages[sortedNumberImages[2]])
        #     # cv2.imwrite("licenseLetters/"+timestamp+"_3.png",numberImages[sortedNumberImages[2]])
        #     # cv2.imshow("fourth number",numberImages[sortedNumberImages[3]])
        #     # cv2.imwrite("licenseLetters/"+timestamp+"_4.png",numberImages[sortedNumberImages[3]])
        #return numberImages[sortedNumberImages[0]]
    
    
    def readLetter(self,img):
        labels = ['0','1','2','3','4','5','6','7','8','9']
        labels.extend(list(string.ascii_uppercase))
        dictionary = {"image" : [] , "vector": [], "label": []}
        grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img_aug = np.repeat(grayImg[..., np.newaxis], 3, -1)
        img_aug = cv2.resize(img_aug,(18,22))
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
    #find the boundaries of the license plate using connected component analysis
    def boundary_finder(self,img,binaryImg):
        #Find connected components
        labelImg = measure.label(binaryImg)
        recImg = img.copy()
        plate_objects_cordinates = []
        plate_like_objects = []
        regions = []
        # regionprops creates a list of properties of all the labelled regions
        #Krysten - changed this because area of 400 was too small for parking numbers
        for region in regionprops(labelImg):
            if region.area < 50 or region.area > 1000:
            #if the region is so small then it's likely not a license plate
                continue

            regions.append(region)
            min_row, min_col, max_row, max_col = region.bbox
            rectBorder = cv2.rectangle(recImg, (min_col, min_row), (max_col, max_row), (0,255,0), 2)
            print("I found a rectangle!")
        
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
        cv2.imshow("grayImg",grayImg)
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












