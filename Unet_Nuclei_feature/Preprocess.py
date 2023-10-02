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
from PIL import Image, ImageFilter

#### Tiles jressica, split, after you need to feed to output.py
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



#DatapredictPath = 'C:/Users/Laura/AppData/Local/Programs/Python/Python36/Phenotype/Unet_nuclei/data'
#TilesPath ='C:/Users/USUARIO/AppData/Local/Programs/Python/Python36/Phenotype/NucleiMaskCNN/WholeSLide/deephistopath/wsi/tiles_png/002'
path_img = 'C:/Users/Laura/AppData/Local/Programs/Python/Python36/Phenotype/Images/001'
path_tiles = 'C:/Users/Laura/AppData/Local/Programs/Python/Python36/Phenotype/Tiles_Glands/001'

#final_pTH='E:/These/NucleiMaskCNN/NucleiMaskCNN/WholeSLide/deephistopath/wsi/tiles_png_hed'
indice_slide=1

list_img = os.listdir(path_img)
print(list_img)

for annot_num, annotation_tif in (enumerate(list_img)):
    imagepath =path_img+'/'+ annotation_tif
    print(annotation_tif)
    img = cv2.imread(imagepath,-1)
    kernel = np.ones((5,5), np.uint8)
    img_dilation = cv2.erode(img, kernel, iterations=1)
    img_dilation= cv2.resize(img, (364,364), interpolation = cv2.INTER_AREA)

    p= os.path.basename(annotation_tif)
    name1 = os.path.splitext(p)[0]
    #cv2.imshow('',img_dilation)
   # cv2.waitKey(0)
   # fname = name1 + '.png'
    #zoomed = zoom(img, 4)
    #fnameIm = os.path.join(path_tiles,fname)
    #cv2.imwrite(fnameIm, zoomed)
    
    print()
    for mm, (img_piece) in enumerate(split_image(img )):
            #print(np.shape(img_piece))
          name=name1+'_'+str(mm)   
          fname = name + '.png'
           #  print(fname)
          fnameIm = os.path.join(path_tiles,fname)
          print(fnameIm)
          zoomed = zoom(img_piece, 3)
          
         # cv2.imshow('',zoomed)
         # cv2.waitKey(0)
          cv2.imwrite(fnameIm, zoomed)
            
            
        
