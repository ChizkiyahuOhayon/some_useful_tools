import os
import json
import numpy as np
import glob
import shutil
import cv2
from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET

import numpy as np

np.random.seed(41)

classname_to_id = {
    "person": 0,
    "bus": 1,
    "car": 2,
    "motorbike": 3,
    "bicycle": 4
}
 
class VOC2CoCo:

    def __init__(self):
        self.images = []
        self.annotations = []
        self.categories = []
        self.img_id = 0
        self.ann_id = 0

    def save_coco_json(self, instance, save_path):
        json.dump(instance, open(save_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)  # indent=2 更加美观显示

    def to_coco(self, xml_path_list):
        self._init_categories()
        for json_path in xml_path_list:
            objInfo = self.read_xmlfile(json_path)
            imgfile = json_path.replace("Annotations","JPEGImages").replace(".xml",".png")
            self.images.append(self._image( imgfile))
            for shape in objInfo:
                annotation = self._annotation(shape)
                self.annotations.append(annotation)
                self.ann_id += 1
            self.img_id += 1
        instance = {}
        instance['info'] = 'spytensor created'
        instance['license'] = ['license']
        instance['images'] = self.images
        instance['annotations'] = self.annotations
        instance['categories'] = self.categories
        return instance


    def _init_categories(self):
        for k, v in classname_to_id.items():
            category = {}
            category['id'] = v
            category['name'] = k
            self.categories.append(category)

    def _image(self,  path):
        image = {}
        img_x = cv2.imread(path)
        h, w,c= img_x.shape
        image['height'] = h
        image['width'] = w
        image['id'] = self.img_id
        image['file_name'] = os.path.basename(path)
        return image

    def _annotation(self, shape):
        # print('shape', shape)
        label = shape[-1]
        points = shape[:-1]
        annotation = {}
        annotation['id'] = self.ann_id
        annotation['image_id'] = self.img_id
        annotation['category_id'] = int(classname_to_id[label])
        annotation['segmentation'] = [np.asarray(points).flatten().tolist()]
        annotation['bbox'] = points
        annotation['iscrowd'] = 0
        annotation['area'] = annotation['bbox'][2] * annotation['bbox'][3]
        return annotation
    def read_xmlfile(self, path):
        in_file = open(path, encoding='utf-8')
        tree = ET.parse(in_file)
        root = tree.getroot()
        allinfos = []
        for obj in root.iter('object'):
            cls = obj.find('name').text
            xmlbox = obj.find('bndbox')
            # # COCO的格式： [x1,y1,w,h] 对应COCO的bbox格式
            allinfos.append([int(float(xmlbox.find('xmin').text)),
                             int(float(xmlbox.find('ymin').text)),
                             int(float(xmlbox.find('xmax').text)) - int(float(xmlbox.find('xmin').text)),
                             int(float(xmlbox.find('ymax').text)) - int(float(xmlbox.find('ymin').text)),
                             cls])
        return allinfos

    def _get_box(self, points):
        min_x = min_y = np.inf
        max_x = max_y = 0
        for x, y in points:
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
        return [min_x, min_y, max_x - min_x, max_y - min_y]


if __name__ == '__main__':
    labelme_path = "/root/RTTS/Annotations"
    saved_coco_path = "/root/RTTS/cocodata/"
    print('reading...')

    if not os.path.exists("%s/annotations/" % saved_coco_path):
        os.makedirs("%s/annotations/" % saved_coco_path)
    if not os.path.exists("%s/images/train2017/" % saved_coco_path):
        os.makedirs("%s/images/train2017" % saved_coco_path)
    if not os.path.exists("%s/images/val2017/" % saved_coco_path):
        os.makedirs("%s/images/val2017" % saved_coco_path)

    print(labelme_path + "/*.xml")
    xml_list_path = glob.glob(labelme_path + "/*.xml")
    print('json_list_path: ', len(xml_list_path))
    train_path, val_path = train_test_split(xml_list_path, test_size=0.1, train_size=0.9)
    print("train_n:", len(train_path), 'val_n:', len(val_path))

    l2c_train = VOC2CoCo()
    train_instance = l2c_train.to_coco(train_path)
    l2c_train.save_coco_json(train_instance, '%s/annotations/instances_train2017.json' % saved_coco_path)
    for file in train_path:
        # shutil.copy(file.replace("json", "jpg"), "%scoco/images/train2017/" % saved_coco_path)
        img_name = file.replace("Annotations","JPEGImages").replace(".xml",".png")
        temp_img = cv2.imread(img_name)
        try:
            cv2.imwrite("{}/images/train2017/{}".format(saved_coco_path, img_name.split(os.sep)[-1]), temp_img)
        except Exception as e:
            print(e)
            print('Wrong Image:', img_name )
            continue
        print(img_name + '-->', img_name.replace('png', 'jpg'))

    for file in val_path:
        # shutil.copy(file.replace("json", "jpg"), "%scoco/images/val2017/" % saved_coco_path)
        img_name = file.replace("Annotations","JPEGImages").replace(".xml",".png")
        temp_img = cv2.imread(img_name)
        try:
            cv2.imwrite("{}/images/val2017/{}".format(saved_coco_path, img_name.split(os.sep)[-1]), temp_img)
        except Exception as e:
            print(e)
            print('Wrong Image:', img_name)
            continue
        print(img_name + '-->', img_name.replace('png', 'jpg'))

    l2c_val = VOC2CoCo()
    val_instance = l2c_val.to_coco(val_path)
    l2c_val.save_coco_json(val_instance, '%s/annotations/instances_val2017.json' % saved_coco_path)

