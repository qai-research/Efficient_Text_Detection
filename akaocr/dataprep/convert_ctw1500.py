import os
import pathlib
import shutil
import json
from PIL import Image
from bs4 import BeautifulSoup
from pathlib import Path

img_path = 'D:/kimnh3/dataset/ocr/raw_download/SCUT-CTW1500/v1-xml/train-1000/train_images'
ann_path = 'D:/kimnh3/dataset/ocr/raw_download/SCUT-CTW1500/v1-xml/train-1000/ctw1500_train_labels'
des_path = 'D:/kimnh3/dataset/ocr/converted/scut-ctw1500/train'
data_set = 'train'
def convert():
    des_img_path = os.path.join(des_path, 'images')
    des_jsn_path = os.path.join(des_path, 'annotations')
    for xml in os.listdir(ann_path):
        if xml.endswith('.xml'):
            print(xml)
            ann = {}

            #get data in xml and append to ann
            with open(os.path.join(ann_path, xml), 'r') as f:
                data = f.read()
            bs_data = BeautifulSoup(data, 'xml')
            boxs = bs_data.find_all('box')
            img_name = bs_data.find('image').get('file')

            #get image wdth, hght
            img = Image.open(os.path.join(img_path, img_name))
            wdth, hght = img.size

            #append img prop to ann
            ann["file"] = img_name
            ann["width"] = wdth
            ann["height"] = hght
            ann["depth"] = 1
            #get words
            words = []
            for box in boxs:
                word = {}
                #get and convert coord
                left, top, width, height = float(box.get('left')), float(box.get('top')), float(box.get('width')), float(box.get('height'))
                x1, y1 = left, top
                x2, y2 = x1+width, y1
                x3, y3 = x2, y2+height
                x4, y4 = x3-width, y3
                print(left, top, width, height)

                #get label text
                lbl = box.find_all('label')[0].text

                #append to words
                word["text"] = lbl
                word["x1"] = round(x1,1)
                word["y1"] = round(y1,1)
                word["x2"] = round(x2,1)
                word["y2"] = round(y2,1)
                word["x3"] = round(x3,1)
                word["y3"] = round(y3,1)
                word["x4"] = round(x4,1)
                word["y4"] = round(y4,1)
                word["char"] = []
                words.append(word)
            ann["words"] = words

            #create folder output
            Path(des_img_path).mkdir(parents=True, exist_ok=True)
            Path(des_jsn_path).mkdir(parents=True, exist_ok=True)
            
            #copy img and save json
            src_img = os.path.join(img_path, img_name)
            des_img = os.path.join(des_img_path, img_name)
            des_jsn = os.path.join(des_jsn_path, img_name[:img_name.rfind('.')]+'.json')

            shutil.copyfile(src_img, des_img)
            with open(des_jsn, 'w') as js:
                json.dump(ann, js, indent=4)

            break
print("Done convert set, check at: ", des_img_path)

