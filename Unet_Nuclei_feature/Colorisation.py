import numpy as np
import cv2
import os
from colortransfert import *

target_img= 'C:/Users/USUARIO/AppData/Local/Programs/Python/Python36/Phenotype/Unet_nuclei/codes/TCGA-001-tile-r5-c109-x110598-y4098-w1024-h1024.png'
source_img = 'E:/These/NucleiMaskCNN/NucleiMaskCNN/WholeSLide/deephistopath/wsi/tiles_png'
final_img ='E:/These/NucleiMaskCNN/NucleiMaskCNN/WholeSLide/deephistopath/wsi/tiles_gland'
target = cv2.imread(target_img)
 
indice_slide = 1
for indice_slide in range(1,28):
    suffix = str(indice_slide).zfill(3)
    source_img_path = source_img + '/' + suffix
    final_img_path = final_img+ '/' + suffix
  #  os.makedirs(final_img_path)
    for annot_num, annotation_tif in (enumerate(os.listdir(source_img_path+'/'))):
        imagepath =source_img_path +'/'+ annotation_tif
        img = cv2.imread(imagepath)
        cv2.imshow('', img )
        cv2.waitKey(0)
        img_final = color_transfer (target,img)
        print(annotation_tif)
        
                    
