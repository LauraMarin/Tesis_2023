import os

import cv2
import shutil
import numpy as np
import skimage.color as sk_color
import matplotlib.pyplot as plt
from skimage.color import rgb2hed
from skimage.color import separate_stains, hdx_from_rgb
from matplotlib.colors import LinearSegmentedColormap
import math
from deconvolution import Deconvolution
import skimage.exposure as sk_exposure
from skimage.exposure import rescale_intensity
import skimage.filters as sk_filters
import skimage.morphology as sk_morphology
from color_transfer import color_transfer

def split_image(image):
    """
    Split image into four small pieces.
    """
    imgheight, imgwidth = image.shape[:2]
    height = imgheight//2
    width = imgwidth//2

    for i in range(imgheight//height):
        for j in range(imgwidth//width):
            yield image[i*height:(i+1)*height, j*width:(j+1)*width]



def get_mpl_colormap(cmap):
    

    # Initialize the matplotlib color map
    sm = plt.cm.ScalarMappable(cmap=cmap)

    # Obtain linear color range
    color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:,2::-1]

    return color_range.reshape(256, 1, 3)



def zoom(img, zoom_factor=4):
    return cv2.resize(img, None, fx=zoom_factor, fy=zoom_factor)

path_img = 'C:/Users/Laura/AppData/Local/Programs/Python/Python36/Phenotype/Unet_nuclei/wsi/tiles_png'
final_pTH='C:/Users/Laura/AppData/Local/Programs/Python/Python36/Phenotype/Output/output_hem_tiles'
indice_slide=1

target_img= 'C:/Users/Laura/AppData/Local/Programs/Python/Python36/Phenotype/Unet_nuclei/codes/TCGA-001-tile-r5-c109-x110598-y4098-w1024-h1024.png'
#source_image= cv2.imread(target_img,-1)
#mean, std = lab_mean_std(source_image)



name_folder = '001'
for indice_slide in range(154,159):
    suffix = str(indice_slide).zfill(3)
    for annot_num, annotation_tif in (enumerate(os.listdir(path_img+ '/'+suffix ))):
        print(annotation_tif)
        print(annot_num)
        imagepath =path_img+'/'+suffix +'/'+ annotation_tif
        print(imagepath)

        img = cv2.imread(imagepath,cv2.COLOR_BGR2RGB)
        print(np.shape(img))


      #  img = reinhard (img,mean, std)
        cmap_hema = LinearSegmentedColormap.from_list('mycmap', ['white', 'navy'])
        ihc_hed = rgb2hed(img)
        hed =(sk_exposure.rescale_intensity(ihc_hed, out_range=(0, 255))).astype("uint8")
        
       # hed = sk_exposure.rescale_intensity(ihc_hed , out_range=(0.0, 1.0))
        equ = sk_exposure.equalize_adapthist(hed[:, :, 2], nbins=256, clip_limit=0.01)
        adapt_equ = (equ * 255).astype("uint8")
        image_bgr = cv2.applyColorMap(adapt_equ, get_mpl_colormap(cmap_hema))
   
        p= os.path.basename(annotation_tif)
        name1 = os.path.splitext(p)[0]
        fname = name1 + '.png'
      
        Image_Name_Path = os.path.join(final_pTH, fname)
        print('hello')
        print(Image_Name_Path)
      #  cv2.imshow('',image_bgr)
       # cv2.waitKey(0)
       # zoomed = zoom(image_bgr, 4)
        cv2.imwrite(Image_Name_Path, image_bgr)
    indice_slide = indice_slide +1


       # for mm, (img_piece) in enumerate(split_image(image_bgr )):
            #print(np.shape(img_piece))
        #     name=name1+'_'+str(mm)   
        #     fname = name + '.png'
           #  print(fname)
         #    fnameIm = os.path.join(final_pTH,fname)
         #    zoomed = zoom(img_piece, 3)
             
          #   print(fnameIm)
          #   cv2.imwrite(fnameIm, zoomed)
            
            
        
