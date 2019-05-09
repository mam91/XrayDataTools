import numpy as np
import os
import shutil

ROOT_DIR = 'C:/Users/mmill/Documents/PythonProjects/XrayDataTools'

def doesFileExist(file_path):
    return os.path.isfile(file_path)

def delete_dir_contents(subdir_to_delete):
    working_dir = os.path.join(ROOT_DIR, subdir_to_delete)
    deleted_count = 0

    for curr_file in os.listdir(working_dir):
        file_to_delete = working_dir + curr_file
        os.remove(file_to_delete)
        deleted_count += 1

    print("Deleted " + str(deleted_count) + " files in " + working_dir)

def copy_data_subset(from_path, to_path, move_percent):
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

        if not doesFileExist(from_img_path):
            print(file_name + " does not exist!")
            continue

        shutil.copyfile(from_mask_path, to_mask_path)
        shutil.copyfile(from_img_path, to_img_path)
    print("Copy complete")