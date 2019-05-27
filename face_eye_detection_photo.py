#!/usr/bin/env python
# coding: utf-8

# In[2]:


# import libraries of python OpenCV  
# where its functionality resides 
import cv2
import matplotlib.pyplot as plt


# In[3]:


#read images from directories
img1 = cv2.imread('C:/Users/Faiz Khan/Desktop/ddd/1.jpg',0)
img2 = cv2.imread('C:/Users/Faiz Khan/Desktop/ddd/2.jpg',0)
img3 = cv2.imread('C:/Users/Faiz Khan/Desktop/ddd/3.jpg',0)
img4 = cv2.imread('C:/Users/Faiz Khan/Desktop/ddd/4.jpg',0)


# In[4]:


plt.imshow(img2)


# In[5]:


plt.imshow(img3)


# In[6]:


# Trained XML classifiers describes some features of some 
# object we want to detect a cascade function is trained 
# from a lot of positive(faces) and negative(non-faces) 
# images. 
face_cascade = cv2.CascadeClassifier('F:/dd/Library/etc/haarcascades/haarcascade_frontalface_alt2.xml')
def detect_face(img):
    face_img = img.copy()
    face_rects = face_cascade.detectMultiScale(face_img)
    for(x,y,w,h) in face_rects:
        cv2.rectangle(face_img,(x,y),(x+w,y+h),(255,255,255),5)
        return face_img


# In[7]:


#perform facial detection
result1 = detect_face(img2)


# In[8]:


plt.imshow(result1)


# In[9]:


# Trained XML file for detecting eyes 
eye_cascade = cv2.CascadeClassifier('F:/dd/Library/etc/haarcascades/haarcascade_eye.xml')
def detect_eyes(img):
    face_img = img.copy()
    eyes_rects = eye_cascade.detectMultiScale(face_img,scaleFactor=1.2,minNeighbors=5)
    for(x,y,w,h) in eyes_rects:
        cv2.rectangle(face_img,(x,y),(x+w,y+h),(255,255,255),2)
        return face_img


# In[10]:


#perform eye detection
result2 = detect_eyes(img2)


# In[11]:


plt.imshow(result2,cmap='gray')


# In[ ]:





# In[ ]:




