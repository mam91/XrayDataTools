import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import shutil
import cv2
import base64
import zlib
import io
from PIL import Image

#   This function provides the functionality to convert from a binary mask to a string of bytes for easy storage
def mask_2_base64(mask):
    img_pil = Image.fromarray(np.array(mask, dtype=np.uint8))
    img_pil.putpalette([0,0,0,255,255,255])
    bytes_io = io.BytesIO()
    img_pil.save(bytes_io, format='PNG', transparency=0, optimize=0)
    bytes = bytes_io.getvalue()
    return base64.b64encode(zlib.compress(bytes)).decode('utf-8')

#   This function provides the functionality to convert from the binary mask bytes back into a binary mask
def base64_2_mask(s):
    z = zlib.decompress(base64.b64decode(s))
    n = np.fromstring(z, np.uint8)
    mask = cv2.imdecode(n, cv2.IMREAD_UNCHANGED)[:, :, 3].astype(bool)
    return mask

#   This function converts the vgg singular file format into a 1-1 image-annotation file format.
def granularize_annotations_from_vgg(vgg_ann_file, input_dir, output_dir):
	annotations = json.load(open(vgg_ann_file))

	for mask in annotations:
		image_file = annotations[mask]['filename']
		mask_file = image_file[:-4] + '.json'

		mask_exists = False

		for data_file in os.listdir(input_dir):
			if mask_file == data_file:
				mask_exists = True
				break

		if not mask_exists:
			image_path = os.path.join(input_dir, image_file)
			image = skimage.io.imread(image_path)
			height, width = image.shape[:2]

			annotations[mask]['file_attributes']['height'] = height
			annotations[mask]['file_attributes']['width'] = width

			annotation = { image_file : annotations[mask] }
			with open(output_dir + mask_file, 'w') as outfile:  
				json.dump(annotation, outfile)

def delete_key(annotation, key):
	try:
		del annotation[key]
	except KeyError:
		pass

#   This function exports images of masks centered with excess space trimmed
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