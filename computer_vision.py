import os
import cv2 as cv
import matplotlib.pyplot as mlt
import numpy as np

os.chdir("S:/Data science/vision")

facecascade = cv.CascadeClassifier("C:/Users/Siddharth/Anaconda_3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")
eyecascade = cv.CascadeClassifier("C:/Users/Siddharth/Anaconda_3/Lib/site-packages/cv2/data/haarcascade_eye.xml")
img = cv.imread("classimg.jpg")
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
faceread = facecascade.detectMultiScale(img, 1.1, 4)

for (x,y,w,h) in faceread:
    cv.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyesread = eyecascade.detectMultiScale(roi_gray)
    
for (ex,ey,ew,eh) in eyesread:
    cv.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), (0,255,0), 2)

mlt.imshow(img)

cv.imwrite('detected_face.jpg',img)