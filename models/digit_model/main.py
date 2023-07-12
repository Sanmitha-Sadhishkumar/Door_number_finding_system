import torch
import PIL.Image as Image
import PIL.ImageEnhance as ImageEnhance
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
from math import *

model = torch.hub.load('icns-distributed-cloud/yolov5-svhn', 'svhn').fuse().eval()
model = model.autoshape()
img1= Image.open(r"C:\Users\Sanmitha\Documents\third.jpg")
# Prediction
h,w=img1.size
pred=[]
filter1=ImageEnhance.Color(img1)
img1=filter1.enhance(0.0)
img1=img1.resize((h*10,w*10))
img1.show()

def predict(i,j,h,w):
    cropped_img = img1.crop((i, j, i+h, j+w))
    f=model(cropped_img)
    if f[0]!=None and f[0][0][4]<=0.4:
        predict(i,j-w,h,w)
        predict(i,j,h,w+w)
        predict(i-h,j,h,w)
        predict(i,j,h*2,w)
    if f[0]!=None and f[0][0][4]>=0.4:
        img2=np.array(cropped_img)
        cv2.rectangle(img2,(round(f[0][0][0].item()),round(f[0][0][1].item())),(round(f[0][0][2].item()),round(f[0][0][3].item())),(255,100,0),6)
        plt.imshow(img2)
        plt.title('class : '+str(f[0][0][5].item())+" \n confidence : "+str(f[0][0][4].item()))
    return f

for i in range(0,h*10,h):
    for j in range(0,w*10,w):
        f=predict(i,j,h,w)
        if f!=None:
            pred.append(f)

for i in pred:
    if i[0]!=None:
        for x1, y1, x2, y2, conf, clas in i[0]:
            if conf>0.3:
                print('box: ({}, {}), ({}, {})'.format(x1, y1, x2, y2))
                print('confidence : {}'.format(conf))
                print('class: {}'.format(int(clas)))
                '''
                print((int(x1/10),int(y1/10)),(int(x2/10),int(y2/10)))
                cv2.rectangle(img2,(int(x1/10),int(y1/10)),(int(x2/10),int(y2/10)),(255,0,0),3)
                plt.imshow(img2)
                plt.show()
                '''