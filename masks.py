import os
import sys
import json
import datetime
import numpy as np
import skimage.draw

ROOT_DIR = os.path.abspath("../../")

def load_shuriken(dataset_dir, subset):
	assert subset in ["train", "val"]
	dataset_dir = os.path.join(dataset_dir, subset)

	for mask_file in os.listdir(dataset_dir):
		if(mask_file.endswith('.json')):
			annotations = json.load(open(os.path.join(dataset_dir, mask_file)))
			annotations = list(annotations.values())
			for a in annotations:
				if type(a['regions']) is dict:
					polygons = [r['shape_attributes'] for r in a['regions'].values()]
				else:
					polygons = [r['shape_attributes'] for r in a['regions']]
				width = (a['file_attributes']['width'])
				height = (a['file_attributes']['height'])

				#self.add_image or whatever goes here
		else:
			continue

def load_mask(iamge_id, height, width, polygons):
	mask = np.zeros([info["height"], info["width"], len(info["polygons"])], dtype=np.uint8)
	for i, p in enumerate(polygons):
		# Get indexes of pixels inside the polygon and set them to 1
		rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
		mask[rr, cc, i] = 1

	# Return mask, and array of class IDs of each instance. Since we have
	# one class ID only, we return an array of 1s
	return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

shuriken_dataset = 'C:/Users/mmill/Downloads/Mask_RCNN-master/datasets/shuriken_opencv'

load_shuriken(shuriken_dataset, 'val')