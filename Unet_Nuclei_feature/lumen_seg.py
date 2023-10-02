from __future__ import division
import cv2
import numpy as np
import os

import matplotlib.pyplot as plt
from skimage.measure import find_contours
from skimage.segmentation import chan_vese
from skimage import img_as_ubyte
from scipy.spatial import Delaunay
import random as rng
import math
import alphashape
from histmatch import hist_match
from skimage.color import separate_stains,fgx_from_rgb
from matplotlib.colors import LinearSegmentedColormap
from skimage.color import rgb2hed
import imutils
from scipy import ndimage
from itertools import combinations
from scipy.spatial import ConvexHull
from skimage.morphology import convex_hull_image
import skimage.io

from skimage.feature import peak_local_max
from skimage.morphology import watershed
from statistics import mean
from sklearn.neighbors import NearestNeighbors

def Delaunay_Point(points_nuclei) :
    tri = Delaunay(points_nuclei,incremental= True)
    shape_points_nuclei = np.shape(points_nuclei)
    pindex = shape_points_nuclei[0]-1
    indice_point = tri.vertex_neighbor_vertices[1][tri.vertex_neighbor_vertices[0][pindex]:tri.vertex_neighbor_vertices[0][pindex+1]]
    points_nuclei_inGland=[ points_nuclei[indice_Delau] for indice_Delau in indice_point]
    #points_nuclei_inGland= np.array(points_nuclei_inGland)
   
    return points_nuclei_inGland


def Roi (roi_Mask, clumen):
    temp = np.zeros(roi_Mask.shape, dtype=np.uint8)
    cv2.fillPoly(temp, pts=[clumen], color=(255, 255, 255))
    masked_image = cv2.bitwise_and(roi_Mask, temp)
    mask_ = cv2.bitwise_not(temp)
    masked_image_ = cv2.bitwise_or(masked_image, mask_)
    return masked_image, masked_image_,temp

def Mean_Value(masked_image, indice):
    locs = np.where(masked_image != indice)
    pixels = masked_image[locs]
    meanValue2 = np.average(pixels, axis=0)
    bwValue = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
    Valuenon = cv2.countNonZero(bwValue)
    meanValue = meanValue2
    return meanValue

def ContourNuclei(img) :
    red_channel = img[:,:,0]
    ret,threshNuclei = cv2.threshold(red_channel,127,255,cv2.THRESH_BINARY_INV) 
    contoursNuclei, hierarchy = cv2.findContours(threshNuclei,cv2.RETR_EXTERNAL ,cv2.CHAIN_APPROX_SIMPLE)
  #  epsilon = 0.1*cv2.arcLength(contoursNuclei[0],True)
  #  approx = cv2.approxPolyDP(contoursNuclei[0],epsilon,True)
    
    return contoursNuclei

                
def center_nuclei(contour_nuclei,cX_nuclei,cY_nuclei,area_thr_Nuclei,indice_nuclei):
    for cnuclei in contour_nuclei:
          if cv2.contourArea(cnuclei)>5:
              indice_nuclei = indice_nuclei+1
              M = cv2.moments(cnuclei)
              cX_nuclei_ = int(M["m10"] / M["m00"])
              cY_nuclei_ = int(M["m01"] / M["m00"])
              cX_nuclei.append(cX_nuclei_)
              cY_nuclei.append(cY_nuclei_)
    return cX_nuclei,cY_nuclei,indice_nuclei



def nuclei_classification(cX_nuclei,cY_nuclei,roi_Glands_Black) :
     for indice_X in range (0,np.size(cX_nuclei)):
               print()
             #  cv2.circle(roi_Glands_Black, (cX_nuclei[indice_X], cY_nuclei[indice_X]), 7, (255, 255, 255), -1)
             #  cv2.imshow('',roi_Glands_Black)
             #  cv2.waitKey(0)
               
               intensity_nuclei = roi_Glands_Black[cY_nuclei[indice_X],cX_nuclei[indice_X]]
               print( intensity_nuclei)
               if intensity_nuclei[0]>100:
                   print(' tumourish cell')
     return intensity_nuclei    

def overlapse_nuclei(masked_image_function,img_nuclei_mask ):
    img_nuclei_ROI = cv2.cvtColor(masked_image_function ,cv2.COLOR_BGR2RGB)
    _, mask2_nuclei= cv2.threshold(masked_image_function  ,0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    res_nuclei = cv2.bitwise_and(img_nuclei_ROI,img_nuclei_ROI, mask=mask2_nuclei)
    hsv2bgr_nuclei = cv2.cvtColor(res_nuclei, cv2.COLOR_HSV2BGR)
    rgb2gray_nuclei = cv2.cvtColor(hsv2bgr_nuclei, cv2.COLOR_BGR2GRAY)
    contours_Nuclei, hierarchy_ = cv2.findContours(rgb2gray_nuclei,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    Nuclei_mask_separated = np.zeros(masked_image_function.shape, dtype="uint8")
    Label_mask_separated = np.zeros(masked_image_function.shape, dtype="uint8")

       ####### still separating overlapsing nuc
    for cnuclei in contours_Nuclei:
        if cv2.contourArea(cnuclei ) > 30 :
            area = cv2.contourArea(cnuclei )
            x_nuclei, y_nuclei, w_nuclei, h_nuclei = cv2.boundingRect(cnuclei)
            masked_single_nuclei = Roi (img_nuclei_mask ,cnuclei)
               
            count_label_mask = np.where((masked_single_nuclei[0] == 255))
               
            D = ndimage.distance_transform_edt(masked_single_nuclei[0])
            localMax = peak_local_max(D, indices=False, min_distance=3,
	labels=masked_single_nuclei[0])
            markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
            labels = watershed(-D, markers, mask=masked_single_nuclei[0])
            Label_mask_separated =Label_mask_separated+labels
          
            for label in np.unique(labels):
                count_label = np.where((labels == label))
                if len(count_label_mask[0]) != 0:
                    ratio_label_mask =  len(count_label[0]) /len(count_label_mask[0])
                else:
                    ratio_label_mask = 0
                   
                if label == 0:
                    continue
                if ratio_label_mask >0.10:
                    if len(count_label[0])>30:
                        mask = np.zeros(masked_single_nuclei[0].shape, dtype="uint8")
                        mask[labels == label] = 255
                        kernel = np.ones((2,2),np.uint8)
                        erosion = cv2.erode(mask,kernel,iterations = 1)
                        Nuclei_mask_separated = Nuclei_mask_separated + erosion
                       
    return Nuclei_mask_separated


def  hierarchy_glands(Roi_lumen_hierar ,contoursMask,c,img_glands_RGB,img_nuclei_mask,perimeter_lumen,area_lumen,indice_lumen,centerLumen_X,centerLumen_Y):
    index = c[1][3]
    masked_imageGlands_hierar = Roi (img_glands_RGB ,c[0])
    masked_imageGlands_Black_hierar = masked_imageGlands_hierar[0]
    masked_imageGlands_White_hierar = masked_imageGlands_hierar[1]
    locs_hierar = np.where(masked_imageGlands_Black_hierar  != 0)
    locs_2_hierar = np.where(masked_imageGlands_Black_hierar > 180)
    if np.size(locs_2_hierar)!=0:
          # print(np.size(locs_hierar)/np.size(locs_2_hierar))
        if np.size(locs_hierar)/np.size(locs_2_hierar) < 5 :
            Roi_lumen_hierar.append(c[0])
            indice_lumen = indice_lumen+1
            M = cv2.moments(c[0])
            cX_lumen_= int(M["m10"] / M["m00"])
            cY_lumen_ = int(M["m01"] / M["m00"])
            centerLumen_X.append(cX_lumen_)
            centerLumen_Y.append(cY_lumen_)
            area_lumen.append(cv2.contourArea(c[0]))
            perimeter_lumen.append(cv2.arcLength(c[0],True))
            masked_hierar = Roi (img_glands_RGB ,contoursMask[index])
            masked_imageGlands_Black_hierar= masked_hierar[1]
      
    return Roi_lumen_hierar,masked_imageGlands_Black_hierar, index,centerLumen_X,centerLumen_Y,perimeter_lumen,area_lumen,indice_lumen

def convex_glands(Roi_lumen_ ,masked_imageGlands,c,img_glands_RGB,centerLumen_X,centerLumen_Y,perimeter_lumen,area_lumen,indice_lumen,masked_imageGlands_Black):
    hull = cv2.convexHull(c[0])
    shape = np.shape(img_glands_RGB)
    mask_convex = np.zeros((shape[0], shape[1]), dtype=np.uint8)
    cv2.drawContours(mask_convex, [hull],-1, (255, 255, 255), -1)
    img_sub = masked_imageGlands[2][...,0]
    background_convex = mask_convex- img_sub
    mask_roi =  cv2.bitwise_and(img_glands_RGB,img_glands_RGB, mask= background_convex)

    convex_contour = cv2.cvtColor(mask_roi ,cv2.COLOR_BGR2GRAY)
    _, bw_convex = cv2.threshold(convex_contour,0,255,cv2.THRESH_BINARY)
    contours_convex, hierarchy_Glands = cv2.findContours(bw_convex, cv2.RETR_EXTERNAL ,cv2.CHAIN_APPROX_SIMPLE)
    for c_convex in contours_convex:
        if cv2.contourArea(c_convex) > 15:
            convex_background = Roi (img_glands_RGB ,c_convex)
            roi_convex_back = convex_background[0]
            locs = np.where(roi_convex_back != 0)
            locs_2 = np.where(roi_convex_back > 180)
             #  print(np.size(locs)/np.size(locs_2))
            if np.size(locs_2)!=0 :
                if np.size(locs)/np.size(locs_2) < 2.2:
                    Roi_lumen_.append( c_convex)
                    indice_lumen = indice_lumen+1
                    M = cv2.moments(c_convex)
                    cX_lumen_ = int(M["m10"] / M["m00"])
                    cY_lumen_ = int(M["m01"] / M["m00"])
                    centerLumen_X.append(cX_lumen_)
                    centerLumen_Y.append(cY_lumen_)
                    area_lumen.append(cv2.contourArea(c_convex))
                    perimeter_lumen.append(cv2.arcLength(c[0],True))
                    masked_imageGlands_Black = masked_imageGlands_Black +  roi_convex_back
   
    return  Roi_lumen_,masked_imageGlands_Black,centerLumen_X,centerLumen_Y,perimeter_lumen,area_lumen,indice_lumen   


def contour_thres (img):
      img_to_contour = cv2.cvtColor(img ,cv2.COLOR_BGR2RGB)
      _, mask2_to_contour= cv2.threshold(img  ,0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
      res_to_contour = cv2.bitwise_and( img_to_contour , img_to_contour , mask=mask2_to_contour)
      hsv2bgr_to_contour = cv2.cvtColor(res_to_contour, cv2.COLOR_HSV2BGR)
      rgb2gray_to_contour = cv2.cvtColor(hsv2bgr_to_contour, cv2.COLOR_BGR2GRAY)
      contours_to_contour, hierarchy_to_contour = cv2.findContours(rgb2gray_to_contour,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
      return  contours_to_contour

def convex_glands_final (img_glands_RGB,Points_convex, Roi_lumen_only_,indice_numberLumen):
    shape = np.shape(img_glands_RGB)
    Points_convex= np.array( Points_convex)
    Roi_lumen_only_[indice_numberLumen] = np.array(  Roi_lumen_only_[indice_numberLumen] )
    mask_convex_del = np.zeros((shape[0], shape[1]), dtype=np.uint8)
    cv2.drawContours(mask_convex_del, [Points_convex],-1, (255, 255, 255), -1)
    cv2.drawContours(mask_convex_del, Roi_lumen_only_,-1, (255, 255, 255), -1)
    contour_hull = contour_thres (mask_convex_del)
    for c_hull in  contour_hull  :
        hull_delaunay = cv2.convexHull(c_hull )
        shape = np.shape(img_glands_RGB)
        mask_total= np.zeros((shape[0], shape[1]), dtype=np.uint8)
        cv2.drawContours(mask_total, [ hull_delaunay],-1, (255, 255, 255), -1)
    return mask_total



def ratio_nuclei(contour_nuclei_final_,Nuclei_mask_separated,img_glands_RGB):
     indice_nuclei_healthy = 0
     indice_nuclei_tumor = 0
     indice_nuclei_total = 0
     Ratio_nuclei_tumor = 0
     Ratio_nuclei_healthy = 0
     for c_nuclei_final  in contour_nuclei_final_:
         indice_nuclei_total = indice_nuclei_total +1
         masked_image_nuclei_final_indi = Roi (Nuclei_mask_separated,c_nuclei_final)
         rgb_nuclei =cv2.bitwise_and(img_glands_RGB,img_glands_RGB, mask=masked_image_nuclei_final_indi[0])
         value = Mean_Value(rgb_nuclei,0)
         if value > 65:
             indice_nuclei_tumor = indice_nuclei_tumor+1
         else:
             indice_nuclei_healthy = indice_nuclei_healthy+1
         Ratio_nuclei_healthy = (indice_nuclei_healthy /indice_nuclei_total)*100
         Ratio_nuclei_tumor= (indice_nuclei_tumor /indice_nuclei_total)*100

     return Ratio_nuclei_tumor, Ratio_nuclei_healthy

def orga_nuclei (points_nuclei,indice_nuclei,cX_nuclei,cY_nuclei):
    for indice_NumberNuclei in range (0,indice_nuclei):
        centerNuclei_XDistance= cX_nuclei[indice_NumberNuclei]
        centerNuclei_YDistance= cY_nuclei[indice_NumberNuclei]
        points_nuclei_ = [centerNuclei_XDistance,centerNuclei_YDistance]
        points_nuclei.append(points_nuclei_)
    return points_nuclei

def reject_outliers_2(data, m=2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / (mdev if mdev else 1.)
    data_range = np.arange(len(data))
    idx_list = data_range[s>=m]
    return data[s < m],idx_list


def delete_weird (points_nuclei):
    nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(points_nuclei)
    distances, indices = nbrs.kneighbors(points_nuclei)
    dis, indice_outliner = reject_outliers_2(distances[:,2], m=10)
    if len(indice_outliner) != 0:
        for ele in sorted(indice_outliner, reverse = True):  
            del points_nuclei[ele] 
   
    return points_nuclei
                   
                    
def organize_point(points_nuclei,points_lumen):
    points_nuclei.append(points_lumen)
    points_nuclei= np.array(points_nuclei)
    points_nuclei_inGland = Delaunay_Point(points_nuclei)
    points_nuclei_inGland.append(points_lumen)
    points_nuclei_inGland= np.array(points_nuclei_inGland)
    return points_nuclei_inGland
    
    
def segmen_glands_final(points_nuclei_inGland ,Points_convex,img_glands_RGB,Roi_lumen_only_,indice_numberLumen):
    cv = ConvexHull(points_nuclei_inGland )
    hull_points = cv.vertices
    grid = np.zeros(img_glands_RGB.shape[:2])
    for indice_hull_points in hull_points :
        Points_convex.append(points_nuclei_inGland[indice_hull_points ])
    Points_convex = np.array(Points_convex )
    cv2.fillPoly(grid, pts=[Points_convex], color=(255, 255, 255))
    image_glands_final = convex_glands_final  (img_glands_RGB,Points_convex, Roi_lumen_only_,indice_numberLumen)
    indices_grid = np.where(image_glands_final == [255])
    coordinates = zip(indices_grid[0], indices_grid[1])
    return coordinates, image_glands_final

def slic_glands (coordinates,label_segment,img_glands_RGB,segments):
     for indice_nuclei_Delaunay in coordinates:
         label_nuclei = segments[indice_nuclei_Delaunay[0],indice_nuclei_Delaunay[1]]
         label_segment.append(label_nuclei)
     label_segmen_list = list(set(label_segment))
     mask_segment = np.zeros(img_glands_RGB.shape[:2], dtype = "uint8")
     for  segVal in label_segmen_list:
         mask_segment[segments == segVal] = 255
         mask_segment_RGB = cv2.bitwise_and(img_glands_RGB, img_glands_RGB, mask =mask_segment)
     return mask_segment,mask_segment_RGB
 

def contour_thres_left (img):
      img_to_contour = cv2.cvtColor(img ,cv2.COLOR_BGR2RGB)
      _, mask2_to_contour= cv2.threshold(img  ,30, 240, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
      res_to_contour = cv2.bitwise_and( img_to_contour , img_to_contour , mask=mask2_to_contour)
      hsv2bgr_to_contour = cv2.cvtColor(res_to_contour, cv2.COLOR_HSV2BGR)
      rgb2gray_to_contour = cv2.cvtColor(hsv2bgr_to_contour, cv2.COLOR_BGR2GRAY)
      contours_to_contour, hierarchy_to_contour = cv2.findContours(rgb2gray_to_contour,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
      return  contours_to_contour

def indice_nuclei_ratio (Ratio_nuclei_tumor,indice_Ratio_nuclei_tumor_0,indice_Ratio_nuclei_tumor_20,indice_Ratio_nuclei_tumor_40,indice_Ratio_nuclei_tumor_60,indice_Ratio_nuclei_tumor_80,indice_Ratio_nuclei_tumor_100):
    if Ratio_nuclei_tumor == 0:
        indice_Ratio_nuclei_tumor_0=indice_Ratio_nuclei_tumor_0+1
    if Ratio_nuclei_tumor>0 and Ratio_nuclei_tumor <= 20:
        indice_Ratio_nuclei_tumor_20=indice_Ratio_nuclei_tumor_20+1
    if Ratio_nuclei_tumor > 20 and Ratio_nuclei_tumor<= 40:
        indice_Ratio_nuclei_tumor_40=indice_Ratio_nuclei_tumor_40+1
    if Ratio_nuclei_tumor>40 and Ratio_nuclei_tumor <= 60:
        indice_Ratio_nuclei_tumor_60=indice_Ratio_nuclei_tumor_60+1
    if Ratio_nuclei_tumor >60 and Ratio_nuclei_tumor <= 80:
        indice_Ratio_nuclei_tumor_80=indice_Ratio_nuclei_tumor_80+1
    if Ratio_nuclei_tumor == 100:
        indice_Ratio_nuclei_tumor_100=indice_Ratio_nuclei_tumor_100+1
    return indice_Ratio_nuclei_tumor_0,indice_Ratio_nuclei_tumor_20,indice_Ratio_nuclei_tumor_40,indice_Ratio_nuclei_tumor_60,indice_Ratio_nuclei_tumor_80,indice_Ratio_nuclei_tumor_100

def full_indice (contour_glands_final_no_lumen, indice_1,indice_2,indice_3,indice_4,indice_5,indice_Area_contour_below_500,indice_Area_contour_below_1500,indice_Area_contour_below_5000,indice_Area_contour_below_10000,indice_Area_contour_below_15000,indice_Area_contour_below_20000 ) :
    if contour_glands_final_no_lumen <=indice_1:
        indice_Area_contour_below_500 = indice_Area_contour_below_500+1
    if contour_glands_final_no_lumen>indice_1 and  contour_glands_final_no_lumen<= indice_2:
        indice_Area_contour_below_1500 = indice_Area_contour_below_1500+1
    if contour_glands_final_no_lumen>indice_2 and contour_glands_final_no_lumen <=indice_3:
        indice_Area_contour_below_5000 = indice_Area_contour_below_5000+1
    if contour_glands_final_no_lumen>indice_3 and contour_glands_final_no_lumen <= indice_4:
        indice_Area_contour_below_10000 = indice_Area_contour_below_10000+1
    if contour_glands_final_no_lumen> indice_4 and contour_glands_final_no_lumen<= indice_5:
        indice_Area_contour_below_15000 = indice_Area_contour_below_15000+1
    if contour_glands_final_no_lumen >indice_5:
        indice_Area_contour_below_20000 = indice_Area_contour_below_20000+1
    return  indice_Area_contour_below_500,  indice_Area_contour_below_1500 ,indice_Area_contour_below_5000,indice_Area_contour_below_10000,indice_Area_contour_below_15000 ,indice_Area_contour_below_20000           
