

from glob import glob                                                           
import cv2
import os
import util
from util import Time
import filter
import tiles
import slide
from PIL import Image

slide.singleprocess_training_slides_to_images()
filter.singleprocess_apply_filters_to_images()
tiles.singleprocess_filtered_images_to_tiles()

