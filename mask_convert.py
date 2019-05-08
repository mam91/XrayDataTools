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

data = "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"
mask = base64_2_mask(data)
print(mask.shape)
print(mask)