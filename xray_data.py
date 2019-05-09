import os
import json

class ImageData():
    def __init__(self):
        self.label = ''
        self.id = ''
        self.path = ''
        self.width = 0
        self.height = 0
        self.bboxes = []

class BBox():
    def __init__(self, class_id, class_label, min_x, min_y, max_x, max_y):
        self.class_id = class_id
        self.class_label = class_label
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y

class XrayDataset():
    def __init__(self):
        self.images = []
        self.classes = {}

    def load_data(self, dataset_dir):
        for mask_file in os.listdir(dataset_dir):
            if(mask_file.endswith('.json')):
                mask_path = os.path.join(dataset_dir, mask_file)
                annotations = json.load(open(mask_path))
        
                file_name = list(annotations.keys())[0]
                annotations = annotations[file_name]

                image_id = mask_file.replace(".json",".png")
                image_path = os.path.join(dataset_dir, image_id)
                label = annotations['label']
                height = annotations['file_attributes']['height']
                width = annotations['file_attributes']['width']

                #if segm mask then get polygons 
                polygons = annotations['regions'][0]['shape_attributes']
                #get bbox from segm mask
                max_y = int(round(max(polygons['all_points_y'])))
                min_y = int(round(min(polygons['all_points_y'])))
                max_x = int(round(max(polygons['all_points_x'])))
                min_x = int(round(min(polygons['all_points_x'])))

                if label not in self.classes:
                    self.classes[label] = len(self.classes.keys()) + 1

                bbox = BBox(self.classes[label], label, min_x, min_y, max_x, max_y)
               
                self.add_image(label,
                    image_id = image_id,
                    path=image_path,
                    width=width, 
                    height=height,
                    bbox = bbox)
            else:
                continue

    def add_image(self, label, image_id, path, width, height, bbox):
        image = ImageData()
        image.label = label
        image.id = image_id
        image.path = path
        image.height = height
        image.width = width
        image.bboxes.append(bbox)
        self.images.append(image)

def main():
    dset_dir = './input'
    dset = XrayDataset()
    dset.load_data(dset_dir)
    print(dset.classes)

if __name__ == '__main__':
    main()