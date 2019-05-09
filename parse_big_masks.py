import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import shutil
import cv2

def strip_masks(directory):
	mask_path = 'C:/Users/mmill/Downloads/Mask_RCNN-master/datasets/shuriken_contours_plus/' + directory + '/'

	annotations = json.load(open(mask_path + 'via_region_data.json'))

	for mask in annotations:
		image_file = annotations[mask]['filename']
		mask_file = image_file[:-4] + '.json'

		mask_exists = False

		for data_file in os.listdir(mask_path):
			if mask_file == data_file:
				mask_exists = True
				break

		if not mask_exists:
			image_path = os.path.join(mask_path, image_file)
			image = skimage.io.imread(image_path)
			height, width = image.shape[:2]

			annotations[mask]['file_attributes']['height'] = height
			annotations[mask]['file_attributes']['width'] = width

			annotation = { image_file : annotations[mask] }
			with open(mask_path + mask_file, 'w') as outfile:  
				json.dump(annotation, outfile)

def delete_key(annotation, key):
	try:
		del annotation[key]
	except KeyError:
		pass

def export_centered_mask_images(from_dir, to_dir):
    mask_path = 'C:/Users/mmill/Documents/GitHub/Education/MaskRCnn/datasets/shuriken_gun/train/'
    to_path = "C:/Users/mmill/Documents/GitHub/Education/MaskRCnn/MaskClassification/tf_files/xray_photos/shuriken/"

    for mask_file in os.listdir(mask_path):
        if mask_file.endswith('.json'):
            annotation = json.load(open(mask_path + mask_file))
            file_name = list(annotation.keys())[0]
                
            info = annotation[file_name]
            file_attr = info['file_attributes']
            polygons = info['regions'][0]['shape_attributes']

            #if you get errors, swap widht and height because some of the annottions are wrong
            mask = np.zeros([file_attr["height"], file_attr["width"]], dtype=np.uint8) + 255
            #mask = np.zeros((height, width), dtype=np.uint8) + 255
            rr, cc = skimage.draw.polygon(polygons['all_points_y'], polygons['all_points_x'])
            mask[rr, cc] = 0
            mask = cv2.blur(mask, (4,4))
                
            max_y = int(round(max(polygons['all_points_y'])))
            min_y = int(round(min(polygons['all_points_y'])))
            max_x = int(round(max(polygons['all_points_x'])))
            min_x = int(round(min(polygons['all_points_x'])))
            
            mask = mask[min_y:max_y, min_x:max_x]
                
            cv2.imwrite(to_path + file_name, mask)