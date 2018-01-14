# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 12:22:11 2018

@author:  Sagar Lakhmani 
    
"""
import pandas as pd
import shutil
import cv2
import glob
import numpy as np
path = 'C:\\Users\\REALSYS3\\Desktop\\Jester Action Recognition\\'
data = pd.read_excel(path+'jester-train.xlsx')
image_list = []
frames = []
y_tr = []
img_rows,img_cols = 32,32
num = data.Num
y = data['Action num']
y2 = []
num2 = []

y2[0:119]=y[0:119]
num2[0:119]=num[0:119]
n=120
m=240
for i in range(len(y)-1):
    if y[i+1] != y[i]:
        y2[n:m-1] = y[i+1:i+121]
        num2[n:m-1] = num[i+1:i+121]
        n = n+120
        m = m+120
    
        
#for i in num:
#    src = '\\\\REALSYS2-XPS\jester dataset\\20bn-jester-v1' +'\\'+ str(i)
#    dst = 'C:\\Users\\REALSYS3\\Desktop\\Jester Action Recognition\\Data\\' + str(i)
#    try:
#        shutil.copytree(src,dst)
#    except:
#        print(str(i)+' not found')
#        
#    print(str(i) + ' done')
j = 0
k = 0
path = 'C:\\Users\\REALSYS3\\Desktop\\Jester Action Recognition\\Data\\Train\\'
count = np.zeros(12)
for i in num2:    
    try:
        
        if len(glob.glob(path+str(i)+'\\*.jpg'))>=30 and count[k]<100:
            l = 0
            print(str(j)+': Folder '+str(i)+' found. Data Preprocessing...')
            for frame in glob.glob(path+str(i)+'\\*.jpg'):
                if l >= 30:
                    break
                f = cv2.imread(frame)
                f = cv2.resize(f,(img_rows,img_cols),interpolation=cv2.INTER_AREA)
                gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                frames.append(gray)
                image_list.append(np.array(gray))
                y_tr.append(y2[j])
                l = l + 1
                
            count[k] = count[k] + 1
            j = j+1
        elif count >= 100:
            continue
            
                
    except:
        print(str(j)+': Folder '+str(i)+' not found')
        count[k] = count[k] - 1
        j = j+1
    
    if j%100 == 0:
        k = k+1
        
