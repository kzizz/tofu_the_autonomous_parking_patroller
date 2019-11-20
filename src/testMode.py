#!/usr/bin/env python

from __future__ import print_function

import roslib
import optparse
import numpy as np
import math
import sys
import rospy
from geometry_msgs.msg import Twist
import matplotlib.pyplot as plt
import cv2
import datetime
import operator
import string
import tensorflow as tf
from tensorflow import keras
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from numpy import loadtxt
from keras.models import load_model
import math
import numpy as np
import re
import string
import cv2
import os
import errno
import random
from collections import Counter
from matplotlib import pyplot as plt
from PIL import Image
inputWidth = 18
inputHeight = 22

dataPath = "/home/fizzer/enph353_cnn_lab/realData"
labels = ['0','1','2','3','4','5','6','7','8','9']
labels.extend(list(string.ascii_uppercase))
dictionary = {"image" : [] , "vector": [], "label": []}


model = load_model('/home/fizzer/enph353_ws/src/tofu_img_process/src/modelWithRealData.h5')

def testModel(img,truth):
  grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  img_aug = np.repeat(grayImg[..., np.newaxis], 3, -1)
  img_aug = cv2.resize(img_aug,(inputWidth,inputHeight))
  img_aug = np.expand_dims(img_aug, axis=0)
  y_predict = model.predict(img_aug)[0]
#   print(y_predict)
  plt.imshow(img)
  predictVal = max(y_predict)
#   print(predictVal)
  predictedVal_index = np.where(y_predict == predictVal)[0][0]
  predictedVal = labels[predictedVal_index]
#   groundTruth_index = np.where(datay_train[index] == 1)[0][0]
#   groundTruth = labels[groundTruth_index]
#   print("predicted value:",format(predictedVal))
#   print("ground truth:",format(truth))
  if (predictedVal == truth):
   print("predicted value:",format(predictedVal))
   print("ground truth:",format(truth))

def readLetters():
    pub = rospy.Publisher('chatter', String, queue_size=10)

    




def main(args):

    for file in os.listdir(dataPath):
        if(file.endswith('.png')):
            truth = file[0]
            image = cv2.imread(dataPath + '/' + file) #converts to grey scale
            plt.imshow(image)
            testModel(image,truth)

if __name__ == '__main__':
    # rospy.init_node('img_processor', anonymous=True) #CHECK : Does this needed to be added to a world.launch file somewhere?
    main(sys.argv)
