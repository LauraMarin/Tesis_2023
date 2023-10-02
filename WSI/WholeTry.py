from glob import glob                                                           
import cv2
import os
import util
from util import Time

from PIL import Image
import openpyxl
import xlrd
from openpyxl import Workbook
from openpyxl import load_workbook
import numpy as np
import shutil

def listdirs(path):
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

def listToString(s):  
    str1 = ""   
    for ele in s:  
        str1 += ele     
    return str



filepath="C:/Users/Laura/AppData/Local/Programs/Python/Python36/Phenotype/Unet_nuclei/wsi/TCGA.txt"
path = 'C:/Users/Laura/AppData/Local/Programs/Python/Python36/Phenotype/Unet_nuclei/gdc/'
pathSlide = 'C:/Users/Laura/AppData/Local/Programs/Python/Python36/Phenotype/Unet_nuclei/wsi/training_slides'
cwd = os.getcwd()


file1 = open(filepath,"a")
indice_slide = 157
indice_column = 3
print('a')
list_sub = os.listdir(path)
print(list_sub)

for dirs in list_sub:
    typeFile= '*tiff'
    subdirectoy = os.path.join(path, dirs)
    print('why')
    print(os.listdir(subdirectoy))
    
    text_files = [f for f in os.listdir(subdirectoy) if f.endswith('.svs')]
    print(text_files)
    #text_files1 = [os.path.join(subdirectoy, f) for f in os.listdir(subdirectoy) if f.endswith('.svs')]
    pngs= glob(subdirectoy)
    str1 = ''.join(text_files)
    print('Name file')
    print(str1)
    #img = cv2.imread(text_files)
   # p= os.path.basename(text_files)
    subdirectoyFile = os.path.join(subdirectoy, str1)
   # nameExt=os.path.splitext(text_files)
    print(subdirectoyFile)
   # 
    dest = shutil.move(subdirectoyFile, pathSlide)
    index = 'TCGA'
    suffix = str(indice_slide).zfill(3)
    impath=os.path.join(index + "-" + suffix+'.svs')
    file1.write((str(text_files)))
    file1.write((str(impath)))
     ##print(impath)
    #impathSlidetraning=os.path.join(pathSlide,str1)
    
    os.chdir(pathSlide)
    os.rename(str1,impath)
    os.chdir(cwd)
    indice_column= indice_column+1
    indice_slide= indice_slide+1
               # p= os.path.basename(files)
               # print(os.path.splitext(file)[1])
    




