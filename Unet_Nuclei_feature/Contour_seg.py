import os
import cv2
import numpy as np
import skimage.exposure as sk_exposure
import matplotlib.pyplot as plt
from skimage.io import imshow, imread
from skimage.color import rgb2hsv, hsv2rgb
from skimage import color
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion

low=40
high=60
kernel = np.ones((4,4), np.uint8)

path_image_contour= 'C:/Users/Laura/AppData/Local/Programs/Python/Python36/Phenotype/Paint/001'

FILENAME='C:/Users/Laura/AppData/Local/Programs/Python/Python36/Phenotype/Images/006/TCGA-001-tile-r12-c5-x4096-y11264-w1024-h1024.PNG' #image can be in gif jpeg or png format 
path_image_final = 'C:/Users/Laura/AppData/Local/Programs/Python/Python36/Phenotype/Images/Image_seg'
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#img=cv2.imread(FILENAME)
#imgplot = plt.imshow(img)
#plt.show()

kernel_dil = np.ones((3,3), np.uint8)
 

#def contour_img (path_image_contour) :
    
list_img = os.listdir(path_image_contour)
for annot_num, annotation_tif in (enumerate(list_img)):
   
    imagepath =path_image_contour+'/'+ annotation_tif
    print(annotation_tif)
    img = cv2.imread(imagepath,1)
    p= os.path.basename(annotation_tif)
    name1 = os.path.splitext(p)[0]
    fname = name1 + '.png'
    path_image_final_1 = os.path.join(path_image_final,fname)
    img= cv2.resize(img, (364,364), interpolation = cv2.INTER_AREA)
    cv2.imshow('',img)
    cv2.waitKey(0)
    img = cv2.erode(img, kernel_dil, iterations=1)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  
    img_fin = np.zeros(img.shape, dtype=np.uint8)
   # img_hsv = cv2.dilate(img_hsv, kernel, iterations=1)



    lower_mask = img_hsv [:,:,0] > 90
    upper_mask = img_hsv [:,:,0] < 130
    saturation = img_hsv [:,:,1] > 100

    mask = upper_mask*lower_mask *saturation
    red = img[:,:,0]*mask
    green = img[:,:,1]*mask
    blue = img[:,:,2]*mask
    red_girl_masked = np.dstack((red,green,blue))
    red_girl_masked = cv2.cvtColor(red_girl_masked, cv2.COLOR_BGR2GRAY)
  
    cv2.imshow('',red_girl_masked)
    cv2.waitKey(0)
 
    ret,threshNuclei = cv2.threshold(red_girl_masked,0,255,cv2.THRESH_BINARY) 

         

    contoursNuclei, hierarchy = cv2.findContours(threshNuclei,cv2.RETR_TREE ,cv2.CHAIN_APPROX_SIMPLE)
 #   cv2.drawContours(img ,contoursNuclei, -1, (0,255,0), 1)
 #   cv2.imshow('',img)
 #   cv2.waitKey(0)

    for c in zip(contoursNuclei, hierarchy[0]):
        if cv2.contourArea(c[0]) > 200:
            if c[1][3] != -1:
             
               
                temp = np.zeros(img.shape, dtype=np.uint8)
                cv2.fillPoly(temp, pts=[c[0]], color=(255, 255, 255))
            #    cv2.imshow('',temp)
            #    cv2.waitKey(0)
                masked_image = cv2.bitwise_and(img, temp)
                Mask_black = cv2.bitwise_not(masked_image)
                mask_ = cv2.bitwise_not(temp)
                masked_image_ = cv2.bitwise_or(masked_image, mask_)
                
                temp_1 = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY )
                #image_max = ndi.maximum_filter(masked_image_, size=20, mode='constant')
                dst = cv2.cornerHarris(temp_1,12,13,0.20)
                dst = cv2.dilate(dst,None)
               
                masked_image_shape = (masked_image_[dst>0.01*dst.max()]).shape
                masked_image_[dst>0.01*dst.max()]=[0,0,255]
              #  cv2.imshow('dst',masked_image_)
              #  cv2.waitKey(0)
                print( masked_image_shape[0])
                if masked_image_shape[0]< 290:
                    img_fin = img_fin+temp
                elif len(masked_image_[dst>0.09*dst.max()])<210:
                    img_fin = img_fin+temp
                
               
   # cv2.imshow('',img_fin)
  #  cv2.waitKey(0)
    cv2.imwrite(path_image_final_1, img_fin)
    cv2.imshow('',img_fin)
    cv2.waitKey(0)




   # cv2.drawContours(img ,contoursNuclei, -1, (0,255,0), 1)
    
