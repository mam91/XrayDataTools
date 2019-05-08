import os
import sys
import json
import skimage
import cv2
import numpy as np
import base64
import zlib
import io
from PIL import Image

def base64_2_mask(s):
    z = zlib.decompress(base64.b64decode(s))
    n = np.fromstring(z, np.uint8)
    mask = cv2.imdecode(n, cv2.IMREAD_UNCHANGED)[:, :, 3].astype(bool)
    return mask

def mask_2_base64(mask):
    img_pil = Image.fromarray(np.array(mask, dtype=np.uint8))
    img_pil.putpalette([0,0,0,255,255,255])
    bytes_io = io.BytesIO()
    img_pil.save(bytes_io, format='PNG', transparency=0, optimize=0)
    bytes = bytes_io.getvalue()
    return base64.b64encode(zlib.compress(bytes)).decode('utf-8')

def load_anns():
    dataset_dir = "C:/Users/mmill/Documents/GitHub/Education/MaskRCnn/datasets/supervisely/train/"
    for mask_file in os.listdir(dataset_dir):
        if(mask_file.endswith('.json')):
            skip_ann = False
            mask_path = os.path.join(dataset_dir, mask_file)
            annotations = json.load(open(mask_path))
            
            label = "person"
            height = annotations['size']['height']
            width = annotations['size']['width']
            objects = annotations['objects']

            masks = []
            origins = []

            try:
                for obj in objects:
                    origin = obj['bitmap']['origin']
                    origins.append(origin)
                    mask = base64_2_mask(obj['bitmap']['data'])
                    masks.append(mask.tolist())
            except:
                print("Failed")
                continue

            full_masks = np.zeros([height, width, len(masks)],dtype=np.uint8)
            
            for i, p in enumerate(masks):
                x_offset = origins[i][1]
                y_offset = origins[i][0]
                for (x, y), value in np.ndenumerate(p):
                    if(value == True):
                        full_masks[x + x_offset][y + y_offset][i] = 1
                    
            
            full_masks = full_masks.astype(np.bool)
            #print(full_masks.shape)
            #masks = np.moveaxis(masks, 0, -1)
            class_labels = np.ones([full_masks.shape[-1]], dtype=np.int32)
            if(i > 1):
                print(mask_file)
                return full_masks

full_mask = load_anns()
print(full_mask.shape)
index = full_mask.shape[2]
print(str(index))

for i in range(index):
    mask = full_mask[:,:,i]
    print(mask.shape)