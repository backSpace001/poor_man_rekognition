#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import libraries of python OpenCV  
# where its functionality resides 
import cv2
##import numpy as np
#import matplotlib.pyplot as plt


# In[2]:



# Trained XML classifiers describes some features of some 
# object we want to detect a cascade function is trained 
# from a lot of positive(faces) and negative(non-faces) 
# images. 
face_cascade = cv2.CascadeClassifier('F:/dd/Library/etc/haarcascades/haarcascade_frontalface_default.xml')
def detect_face(img):
    face_img = img.copy()
    face_rects = face_cascade.detectMultiScale(face_img)
    for(x,y,w,h) in face_rects:
        cv2.rectangle(face_img,(x,y),(x+w,y+h),(255,255,255),10)
        return face_img


# In[ ]:





# In[3]:



# Trained XML file for detecting eyes 
eye_cascade = cv2.CascadeClassifier('F:/dd/Library/etc/haarcascades/haarcascade_eye.xml')
def detect_eyes(img):
    face_img = img.copy()
    eyes_rects = eye_cascade.detectMultiScale(face_img)
    for(x,y,w,h) in eyes_rects:
        cv2.rectangle(face_img,(x,y),(x+w,y+h),(255,255,255),10)
        return face_img


# In[ ]:





# In[ ]:





# In[4]:


# capture frames from a camera 
cap = cv2.VideoCapture('C:/Users/Faiz Khan/Pictures/Camera Roll/sample2.mp4') 
  
# loop runs if capturing has been initialized. 
while 1:  
  
    # reads frames from a camera 
    ret, img = cap.read()  
  
    # convert to gray scale of each frames 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
  
    # Detects faces of different sizes in the input image 
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) 
  
    for (x,y,w,h) in faces: 
        # To draw a rectangle in a face  
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)  
        roi_gray = gray[y:y+h, x:x+w] 
        roi_color = img[y:y+h, x:x+w] 
  
        # Detects eyes of different sizes in the input image 
        eyes = eye_cascade.detectMultiScale(roi_gray)  
  
        #To draw a rectangle in eyes 
        for (ex,ey,ew,eh) in eyes: 
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,127,255),2) 
  
    # Display an image in a window 
    cv2.imshow('img',img) 
  
    # Wait for Esc key to stop 
    k = cv2.waitKey(30) & 0xff
    if k == 27: 
        break
  # print("Writing frame {} / {}".format(frame_number, length))
    #output_movie.write(img)

# All done!
input_movie.release()

#Close the window 
cap.release() 
  
# De-allocate any associated memory usage 
cv2.destroyAllWindows()  


# In[ ]:




