#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
from darkflow.net.build import TFNet
import numpy as np
import time


# In[4]:


option = {
    'model': 'cfg/yolo.cfg',
    'load': 'bin/yolov2.weights',
    'threshold': 0.15,
    'gpu': 1.0
}


# In[9]:


tfnet = TFNet(option)

capture = cv2.VideoCapture('sample3.avi')
colors = [tuple(255 * np.random.rand(3)) for i in range(5)]
size = (
    int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
    int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
)
frame1 = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',frame1, 20.0, size)

while (capture.isOpened()):
    stime = time.time()
    ret, frame = capture.read()
    if ret:
        results = tfnet.return_predict(frame)
        for color, result in zip(colors, results):
            tl = (result['topleft']['x'], result['topleft']['y'])
            br = (result['bottomright']['x'], result['bottomright']['y'])
            label = result['label']
            frame = cv2.rectangle(frame, tl, br, color, 7)
            frame = cv2.putText(frame, label, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
            
        out.write(frame)
        cv2.imshow('frame', frame)

        print('FPS {:.1f}'.format(1 / (time.time() - stime)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        capture.release()
        out.release()
        cv2.destroyAllWindows()
        break

