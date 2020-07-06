######################################################################################################################################################
#
# Project Name : Detecting Motion and Converting and Capturing Images
# Description : When motion is detected, it stores the image in a directory
# 
# I Followed pknowledge's basic_motion_detection_opencv_python.py (https://gist.github.com/pknowledge/623515e8ab35f1771ca2186630a13d14) 
# and extended to store the images in the directory (CPD)/result/Images.
#
# It used OPENCV2 (v 4.2.0), PIL and other libraries
#
# Author : Hariraj R
# Date : 13-05-2020 
#
######################################################################################################################################################

# Libraries Used
# if not present you can easily install using pip
# for example if you dont have cv2 installed
# you can simply type
# pip install opencv-python

import cv2
import numpy as np
import os
from PIL import Image
import pandas as pd
import datetime

# creates directory 'results/images' and stores the image
output_directory = 'results'
os.makedirs(output_directory,exist_ok=True)
os.makedirs(output_directory+'/images', exist_ok=True)

# Initialize Webcam Feed
# if USB CAM is connected try with 1,2
cap = cv2.VideoCapture('http://192.168.0.101:4747/video') #0

# Get Frame width and height
frame_width = int( cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int( cap.get( cv2.CAP_PROP_FRAME_HEIGHT))

# Read feed two times and store it in two frames
# If there is a change in frame1 and frame2 it stores the frame1 as output
ret, frame1 = cap.read() 
ret, frame2 = cap.read()

#print(frame1.shape)
# Enable timestamp and img_path 
df = pd.DataFrame(columns=['timestamp', 'img_path'])

# When Camera is Opened
while cap.isOpened():
    diff = cv2.absdiff(frame1, frame2)  # returns abs difference value between frame1 and frame2
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)   # Converts to Gray Scale image
    blur = cv2.GaussianBlur(gray, (5,5), 0) 
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        
        array = cv2.cvtColor(np.array(frame1), cv2.COLOR_RGB2BGR)
        image = Image.fromarray(array)
        file_path = output_directory+'/images/'+str(len(df))+'.jpg' #USING PIL
        
        if cv2.contourArea(contour) < 2000: #if contour area is more than 2000px then store the image
            continue
        image.save(file_path, "JPEG")   
        df.loc[len(df)] = [datetime.datetime.now(), file_path]

    # show feed view
    cv2.imshow("feed", frame1)
    frame1 = frame2
    ret, frame2 = cap.read()
     
    if cv2.waitKey(40) == 27: #escape key
        break
    
cv2.destroyAllWindows()
cap.release()

