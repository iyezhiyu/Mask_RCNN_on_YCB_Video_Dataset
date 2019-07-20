#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 22:58:48 2019

@author: Zhiyu Ye

Email: yezhiyu@hotmail.com

In London, the United Kingdom
"""
"""
Using the synthetic images in the YCB Video Dataset to train a Mask R-CNN.
This file is to generate a json file for annotations as the format in the COCO dataset.
"""

import os
import json
import cv2
import numpy as np
from PIL import Image
from tools.sub_masks_annotations import create_sub_masks, create_sub_mask_annotation
import time
import matplotlib.pyplot as plt



if __name__ == "__main__":
    
    input_dir = 'path to/YCB_Video_Dataset'
    output_dir = 'path to/YCBVD_Datasyn_for_train'
    
    # Generate the categories
    class_file = open(input_dir + '/image_sets/classes.txt')
    line = class_file.readline()
    category_id = 0
    categories = []
    while line:
        category_id += 1
        category = {'supercategory':line, 'id':category_id, 'name':line}
        categories.append(category)
        line = class_file.readline()
    class_file.close()
    
    
    # Generate the images and the annotations
    files = os.listdir(input_dir + '/data_syn')
    width = 640
    height = 480
    iscrowd = 0
    annotation_id = 0
    annotations = []
    images = []
    count = 0
    for file in files:
        if file[-3:] == 'png' and file[7:12] == 'label':
            start_time = time.time()
            print('Processing:', file, '...')
            # Write infomation of each image
            file_name = file[:7] + 'color.png'
            image_id = int(file[:6])
            image_item = {'file_name':file_name, 'height':height, 'id':image_id, 'width':width}
            images.append(image_item)
            
            # Write information of each mask in the image
            image = Image.open(input_dir + '/data_syn/' + file)
            # Extract each mask of the image
            sub_masks = create_sub_masks(image)
            count = count + len(sub_masks)
            for category_id, sub_mask in sub_masks.items():
                category_id = int(category_id[1:category_id.find(',')])
                annotation_id += 1
                cimg = np.array(sub_mask)
                opencvImage =  np.stack((cimg, cimg, cimg), axis = 2)
                instance = np.uint8(np.where(opencvImage == True, 0, 255))
                annotation_item = create_sub_mask_annotation(instance, image_id, category_id, annotation_id, iscrowd)
                annotations.append(annotation_item)
                                
            print('Done! Time used:', time.time()-start_time)
    
    print('Test if all the instances are detected, the result is', count == annotation_id)
    # Combine categories, annotations and images to form a json file
    json_data = {'annotations':annotations, 'categories':categories, 'images':images}
    annotations_output_dir = output_dir + '/annotations'
    if not os.path.exists(annotations_output_dir):
                os.makedirs(annotations_output_dir)
    with open(annotations_output_dir + '/instances.json', 'w') as f:
        json.dump(json_data, f)
