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

def base64_2_mask(s):
    z = zlib.decompress(base64.b64decode(s))
    n = np.fromstring(z, np.uint8)
    mask = cv2.imdecode(n, cv2.IMREAD_UNCHANGED)[:, :, 3].astype(bool)
    return mask

def doesFileExist(file_path):
    return os.path.isfile(file_path)

def delete_cnn_data(root_dataset, subset_deleting):
    from_path = 'C:/Users/mmill/Documents/GitHub/Education/MaskRCnn/MaskClassification/tf_files/' + root_dataset + '/' + subset_deleting + '/'
    deleted_count = 0
    for mask_file in os.listdir(from_path):
        file_to_delete = from_path + mask_file
        os.remove(file_to_delete)
        deleted_count += 1
    print("Deleted " + str(deleted_count) + " files in " + from_path)

def delete_dir_contents(root_dataset, subset_deleting):
    from_path = 'C:/Users/mmill/Documents/GitHub/Education/MaskRCnn/datasets/' + root_dataset + '/' + subset_deleting + '/'
    deleted_count = 0
    for mask_file in os.listdir(from_path):
        file_to_delete = from_path + mask_file
        os.remove(file_to_delete)
        deleted_count += 1
    print("Deleted " + str(deleted_count) + " files in " + from_path)

def doesDataKeyExists(mask_path):
    annotations = json.load(open(mask_path))
    try:
        objects = annotations['objects']
        for obj in objects:
            mask_data = obj['bitmap']['data']
    except:
        print("Failed")
        return False
    return True

def move_training_data(root_dataset, subset_from, subset_to, move_percent):
    from_path = 'C:/Users/mmill/Documents/GitHub/Education/MaskRCnn/datasets/' + root_dataset + '/' + subset_from + '/'
    to_path = 'C:/Users/mmill/Documents/GitHub/Education/MaskRCnn/datasets/' + root_dataset + '/' + subset_to + '/'

    file_list = []
    for mask_file in os.listdir(from_path):
        if mask_file.endswith('.json'):
            file_list.append(mask_file)

    print("Total file count: " + str(len(file_list)))

    num_to_choose = int(round(len(file_list) * move_percent))

    print("Number to choose: " + str(num_to_choose))

    trimmed_file_list = np.random.choice(file_list, num_to_choose, replace=False)

    print("Trimmed file count: " + str(len(trimmed_file_list)))

    for mask_file in trimmed_file_list:
        file_name = mask_file.replace(".json","")

        from_mask_path = from_path + mask_file
        from_img_path = from_path + file_name
        to_mask_path = to_path + mask_file
        to_img_path = to_path + file_name

        if not doesDataKeyExists(from_mask_path):
            continue

        if not doesFileExist(from_img_path):
            print(file_name + " does not exist!")
            continue

        shutil.copyfile(from_mask_path, to_mask_path)
        shutil.copyfile(from_img_path, to_img_path)
    print("Copy complete")

def export_masks():
    dataset_dir = "C:/Users/mmill/Documents/GitHub/Education/MaskRCnn/datasets/supervisely/train/"
    mask_dir = "C:/Users/mmill/Documents/GitHub/Education//MaskRCnn/MaskClassification/tf_files/supervisely/person/"

    for mask_file in os.listdir(dataset_dir):
        if(mask_file.endswith('.json')):
            mask_path = os.path.join(dataset_dir, mask_file)
            annotations = json.load(open(mask_path))
            
            label = "person"
            height = annotations['size']['height']
            width = annotations['size']['width']
            objects = annotations['objects']

            image_name = mask_file.replace(".json","").replace(".png", "")

            masks = []
            try:
                for obj in objects:
                    mask = base64_2_mask(obj['bitmap']['data'])
                    masks.append(mask.tolist())
            except:
                print("Failed")
                continue

            mini_mask = np.zeros([len(masks), height, width],dtype=np.uint8)
            
            for i, p in enumerate(masks):
                mini_mask = np.zeros([height, width],dtype=np.uint8)

                for (x, y), value in np.ndenumerate(p):
                   if(value == True):
                       mini_mask[x][y] = 255
                
                mini_mask = cv2.bitwise_not(mini_mask)

                cv2.imwrite(mask_dir + image_name + str(i) + ".png", mini_mask)

#delete_dir_contents("supervisely", "train")
#move_training_data("supervisely", "staging", "train", 0.40)
delete_cnn_data("supervisely", "person")
export_masks()