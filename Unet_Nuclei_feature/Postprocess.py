
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def combine_images(img_list):
    """ Combines imgs using indexes as follows:
        0 1
        2 3
    """
    
    up = np.hstack(img_list[:4])
    down = np.hstack(img_list[4:8])
    fullFist = np.vstack([up, down])
    upND= np.hstack(img_list[8:12])
    downND = np.hstack(img_list[12:])
    fullND = np.vstack([upND, downND])
    full =np.vstack([fullFist, fullND])
    print(np.shape(full))
   # cv2.imshow('',full)
    #cv2.waitKey(0)
   # plt.show()
  
    return full
image_predic_path = 'C:/Users/USUARIO/AppData/Local/Programs/Python/Python36/Phenotype/Unet_nuclei/output'










for num, name_files in (enumerate(os.listdir(image_predic_path))):
    indice_slide = 0
    print(name_files)
    img_list = []
    
    while indice_slide < 16 :
        suffix = str(indice_slide)
        fname = image_predic_path+'/'+str(name_files)+'/'+str(name_files)+'_'+suffix+'.png'
        print(fname)
        img = cv2.imread(fname)
        print(np.shape(img))
        img = cv2.resize(img, (256,256))
        img_list.append(img)
        indice_slide = indice_slide +1
        #os.remove(fname)
    
    img_full = combine_images(img_list)
   
    full_image_path =image_predic_path +'/'+str(name_files)+'.png'
    cv2.imwrite(full_image_path,img_full)

        
           
