import cv2
import math
import numpy as np
import time
from cvzone.HandTrackingModule import HandDetector

#Importing classifier 
from cvzone.ClassificationModule import Classifier

cap = cv2.VideoCapture(0)   # 0 is the id no for webcam
detector = HandDetector(maxHands=1)

#Adding the classifier



classifier = Classifier("Model/keras_model.h5","Model/labels.txt")

offset = 20
folder = "C:\SIGN LANG\O"

while True:
    success, img = cap.read() 
    hands, img = detector.findHands(img) 
    if hands:           #if hand is detected
        hand = hands[0]
        x,y,w,h = hand['bbox']

        imgWhite = np.ones((300,300,3),np.uint8)*255 #A fixed matrix of 300x300 which we ll put the image in so that all images are the same size (300by300)
        #imgCrop = img[y:y+h,x:x+w] this was cropping the tips of fingers
        imgCrop = img[y-offset:y+h+offset,x-offset:x+w+offset]

        imgCropShape = imgCrop.shape

        #next we are gonna try to resize the image so it fits in the box better

        aspectRatio = h/w
        if aspectRatio > 1:
            k = 300/h      #imgsize/height
            wCalc = math.ceil(k*w) #calculated width
            imgResize = cv2.resize(imgCrop,(wCalc,300)) 
            imgResizeShape = imgResize.shape
            wGap = math.ceil((300-wCalc)/2)
            #imgWhite[0:imgResizeShape[0],0:imgResizeShape[1]] = imgResize
            imgWhite[:,wGap:wCalc+wGap] = imgResize #center aligned

            #CLASSIFICATION

            prediction, index = classifier.getPrediction(imgWhite)
            print(prediction,index)

        else:
            k = 300/w      #imgsize/height
            hCalc = math.ceil(k*h) #calculated width
            imgResize = cv2.resize(imgCrop,(hCalc,300)) 
            imgResizeShape = imgResize.shape
            hGap = math.ceil((300-hCalc)/2)
            #imgWhite[0:imgResizeShape[0],0:imgResizeShape[1]] = imgResize
            imgWhite[:,hGap:hCalc+hGap] = imgResize #center aligned

            #CLASSIFICATION

            prediction, index = classifier.getPrediction(imgWhite)
            print(prediction,index)
            

        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgWhite)

    cv2.imshow('Image', img)
    cv2.waitKey(1)    # 1ms delay

#74 DAYS MORE TO GETTING A LAPTOPPPPP YAYYYYYY
# 148 HOURS TO GO

#day1 - 2hrs 16minutes
#day 2 - 