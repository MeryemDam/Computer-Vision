#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 16:01:11 2018

@author: Meryem
"""


import matplotlib.pyplot as plt
from skimage import io
import numpy as np

## Depth map 


#%% Affichage d'images 



img1 = io.imread('syntheticd.pgm')
img2 = io.imread('syntheticg.pgm')

N,M = img1.shape


plt.figure()
plt.imshow(img1, cmap=plt.cm.gray)
plt.axis('off')
plt.title('Image1')
plt.savefig('image1')
plt.figure()

plt.figure()
plt.imshow(img2, cmap=plt.cm.gray)
plt.axis('off')
plt.title('Image2')
plt.savefig('image2')
plt.figure()


#%% 
    

img_res_1 = np.zeros((N,M))

for i in range (N):
    for j in range (M):        
        d = []
        ssd = 0
        
        for k in range (N):
            ssd+=np.abs((int(img1[i,j])-int(img2[i,k]))**2)
            d.append(ssd)
            
            
        img_res_1[i,j] = np.min(d)


        

plt.figure()
plt.imshow(img_res_1, cmap=plt.cm.gray)
plt.axis('off')
plt.title('Image resultat 1*1 ')
plt.plot()


#%% Windows(3x3)

def Init_img3(img):
    N,M = img.shape
    imgr = np.zeros ((N+2,M+2))
    
    for i in range (N):
        for j in range (M):
            imgr[i+1,j+1] = img[i,j]
    return imgr

imgr1 = Init_img3(img1)
imgr2 = Init_img3(img2)

windows = [(-1,-1),(-1,0),(-1,1),(0,1),(0,-1),(1,1),(1,-1),(1,0),(-1,1)]
img_result_3 = np.zeros((N+1,M+1))
for i in range (1,N+1):
    for j in range (1,M+1):
        d = []
        for k in range (1,M+1):
            ssd = 0
            for w in windows:
                ssd += (imgr1[i+w[0],j+w[1]]-imgr2[i+w[0],k+w[1]])**2
            d.append(ssd)
        img_result_3[i][j] = int(min(d))

plt.figure()
plt.imshow(img_result_3, cmap=plt.cm.gray)
plt.axis('off')
plt.title('Image resultat 3*3 ')
plt.plot()

#%% windows (5x5)

def Init_img5(img):
    N,M = img.shape
    imgr = np.zeros ((N+4,M+4))
    
    for i in range (N):
        for j in range (M):
            imgr[i+2,j+2] = img[i,j]
    return imgr


imgr1 = Init_img5(img1)
imgr2 = Init_img5(img2)

windows = [(i,j) for i in range(-2,3) for j in range(-2,3)]
img_result_5 = np.zeros((N+4,M+4))

for i in range (2,N+2):
    for j in range (2,M+2):
        d = []
        for k in range (2,M+2):
            ssd = 0
            for w in windows:
                ssd += (imgr1[i+w[0],j+w[1]]-imgr2[i+w[0],k+w[1]])**2
            d.append(ssd)
        img_result_5[i][j] = int(min(d))

plt.figure()
plt.imshow(img_result_5, cmap=plt.cm.gray)
plt.axis('off')
plt.title('Image resultat 5*5 ')
plt.show()

#%% windows(7x7)

def Init_img7(img):
    N,M = img.shape
    imgr = np.zeros ((N+6,M+6))
    
    for i in range (N):
        for j in range (M):
            imgr[i+3,j+3] = img[i,j]
    return imgr


imgr1 = Init_img7(img1)
imgr2 = Init_img7(img2)

windows = [(i,j) for i in range(-3,4) for j in range(-3,4)]
img_result_7 = np.zeros((N+6,M+6))

for i in range (3,N+3):
    for j in range (3,M+3):
        d = []
        for k in range (3,M+3):
            ssd = 0
            for w in windows:
                ssd += (imgr1[i+w[0],j+w[1]]-imgr2[i+w[0],k+w[1]])**2
            d.append(ssd)
        img_result_7[i][j] = int(min(d))

plt.figure()
plt.imshow(img_result_7, cmap=plt.cm.gray)
plt.axis('off')
plt.title('Image resultat 7*7 ')
plt.show()


#%%
def SSD_Function (img1, img2, Twindows):
    L,C = img1.shape
    L2,C2 = img2.shape
    imgP1 = np.zeros ((L+Twindows-1,C+Twindows-1))
    imgP2 = np.zeros ((L2+Twindows-1,C2+Twindows-1))
    imgR = np.zeros ((L2+Twindows-1,C2+Twindows-1))
    
    #Prolongation matrice
    b = int((Twindows-1)/2)
    imgP1 [b:L+b, b:C+b] = img1
    imgP2 [b:L2+b, b:C2+b] = img2 
    
    if(Twindows % 2 == 0):
        print("choose another windows ...!")
    else :
        for i in range (b,L):
            for j in range(b,C):
                v = []
                for k in range(b,C2):
                    v.append(np.sum(np.power((imgP1[i-b:i+b+1, j-b:j+b+1] - imgP2[i-b:i+b+1, k-b:k+b+1]),2)))
                imgR[i,j]=min(v)
    return imgR
plt.figure()
plt.imshow(SSD_Function(img1,img2,3), cmap=plt.cm.gray)
plt.axis('off')
plt.title('Image resultat en utilisant la fonction générique ')

#%%Plotting all of the results into one figure
plt.subplot(221)
plt.title('window 1*1')
plt.imshow(img_res_1,cmap=plt.cm.gray)
plt.subplot(222)
plt.title('window 3*3')
plt.imshow(img_result_3,cmap=plt.cm.gray)
plt.subplot(223)
plt.title('window 5*5')
plt.imshow(img_result_5,cmap=plt.cm.gray)
plt.subplot(224)
plt.title('window 7*7')
plt.imshow(img_result_7,cmap=plt.cm.gray)
plt.show()

#%%

plt.figure()
plt.imshow(SSD_Function(img1,img2,11), cmap=plt.cm.gray)
plt.axis('off')
plt.title('Image resultat en utilisant 11*11 ')

#%% Test sur d'autres images Teddy

img3 = io.imread('teddyd.pgm')
img4 = io.imread('teddyg.pgm')

N,M= img3.shape
img_res_1 = np.zeros((N,M))

for i in range (N):
    for j in range (M):        
        d = []
        ssd = 0
        
        for k in range (N):
            ssd+=np.abs((int(img3[i,j])-int(img4[i,k]))**2)
            d.append(ssd)
            
            
        img_res_1[i,j] = np.min(d)


        

plt.subplot(131)
plt.imshow(img3, cmap=plt.cm.gray)
plt.axis('off')
plt.title('Image originale 1 ')
plt.subplot(132)
plt.imshow(img4, cmap=plt.cm.gray)
plt.axis('off')
plt.title('Image originale 2 ')
plt.subplot(133)
plt.imshow(img_res_1, cmap=plt.cm.gray)
plt.axis('off')
plt.title('Disparity Map 1*1 ')
plt.show()

#%%
img5 = io.imread('conesd.pgm')
img6 = io.imread('conesg.pgm')

N,M= img6.shape
img_res_6 = np.zeros((N,M))

for i in range (N):
    for j in range (M):        
        d = []
        ssd = 0
        
        for k in range (N):
            ssd+=np.abs((int(img5[i,j])-int(img6[i,k]))**2)
            d.append(ssd)
            
            
        img_res_6[i,j] = np.min(d)


        

plt.subplot(131)
plt.imshow(img5, cmap=plt.cm.gray)
plt.axis('off')
plt.title('Image originale 1 ')
plt.subplot(132)
plt.imshow(img6, cmap=plt.cm.gray)
plt.axis('off')
plt.title('Image originale 2 ')
plt.subplot(133)
plt.imshow(img_res_6, cmap=plt.cm.gray)
plt.axis('off')
plt.title('Disparity Map 1*1 ')
plt.show()



#%%Cone resize
from scipy.misc import imresize
img5 = imresize(img5,(100,100))
img6 = imresize(img6,(100,100))

plt.subplot(131)
plt.imshow(img5, cmap=plt.cm.gray)
plt.axis('off')
plt.title('im orig 100*100 ')
plt.subplot(132)
plt.imshow(img6, cmap=plt.cm.gray)
plt.axis('off')
plt.title('img 2 100*100 ')
plt.subplot(133)
plt.imshow(SSD_Function(img5,img6,3), cmap=plt.cm.gray)
plt.axis('off')
plt.title('Disparity Map 3*3 ')
plt.show()

#%%
from scipy.misc import imresize
from scipy import ndimage
img5 = io.imread('conesd.pgm')
img6 = io.imread('conesg.pgm')
img5 = imresize(img5,(100,100))
img6 = imresize(img6,(100,100))


plt.subplot(131)
plt.imshow(img5, cmap=plt.cm.gray)
plt.axis('off')
plt.title('im orig 100*100 ')
plt.subplot(132)
plt.imshow(img6, cmap=plt.cm.gray)
plt.axis('off')
plt.title('img 2 100*100 ')
plt.subplot(133)
plt.imshow(ndimage.median_filter(SSD_Function(img5,img6,5),(7,7)), cmap=plt.cm.gray)
plt.axis('off')
plt.title('Disparity Map 7*7 ')
plt.show()
