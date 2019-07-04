#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import pytesseract
from PIL import Image


# In[10]:


img = Image.open('4.png')


# In[11]:


pytesseract.pytesseract.tesseract_cmd = 'C:/Users/Faiz Khan/Desktop/ddd/text recognition/Tesseract-OCR/tesseract'


# In[ ]:





# In[12]:


result = pytesseract.image_to_string(img)


# In[13]:


with open('reader.txt', mode='w') as file:
    file.write(result)
    


# In[ ]:




