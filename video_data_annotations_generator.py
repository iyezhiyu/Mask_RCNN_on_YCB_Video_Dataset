#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 14:42:15 2019

@author: Zhiyu Ye

Email: yezhiyu@hotmail.com

In London, the United Kingdom
"""

import os
import json
import cv2
import numpy as np
from PIL import Image
from tools.sub_masks_annotations import create_sub_masks, create_sub_mask_annotation
import time
import random
import matplotlib.pyplot as plt


if __name__ == "__main__":
    
    input_dir = 'path to/YCB_Video_Dataset'

    # Generate the categories
    class_file = open(input_dir + '/image_sets/classes.txt')
    line = class_file.readline()
    count = 0
    category_id = 0
    categories = []
    while line:
        category_id += 1
        category = {'supercategory':line, 'id':category_id, 'name':line}
        categories.append(category)
        line = class_file.readline()
    class_file.close()
    
    # Read the names of the images to generator annotations
    image_names_file = open(input_dir + '/image_sets/train.txt')
    line = image_names_file.readline()
    image_names = []
    while line:
        image_names.append(line[:-1])
        line = image_names_file.readline()
    image_names_file.close()
    
    # For shuffle the data
    num_of_images = len(image_names)
    random.seed(0)
    image_id_index = random.sample([i for i in range(0, num_of_images)], num_of_images)
    
    # Generate the images and the annotations
    image_dir = input_dir + '/data'
    width = 640
    height = 480
    iscrowd = 0
    annotation_id = 0
    annotations = []
    images = []
    image_count = -1
    count = 0
    for image_name in image_names:
        
        start_time = time.time()
        print('Processing:', image_name, '...')
        
        # Write infomation of each image
        file_name = image_name + '-color.png'
        image_count += 1
        image_id = image_id_index[image_count]
        image_item = {'file_name':file_name, 'height':height, 'id':image_id, 'width':width}
        images.append(image_item)
        
        # Write information of each mask in the image
        mask_name = image_name + '-label.png'
        image = Image.open(image_dir + '/' + mask_name)
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
    annotations_output_dir = input_dir + '/annotations'
    if not os.path.exists(annotations_output_dir):
                os.makedirs(annotations_output_dir)
    with open(annotations_output_dir + '/instances.json', 'w') as f:
        json.dump(json_data, f)
    
    

