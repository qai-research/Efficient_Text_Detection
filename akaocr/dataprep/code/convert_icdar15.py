import numpy as np
from pathlib import Path
import cv2
import shutil
import tqdm
import argparse
import os
from shutil import copy
import json

subset = ['train','test']

def convert(train_img, train_gt, test_img, test_gt, des_path):
    #remove if exist
    count = 0
    for ss in subset:
        dest = Path(des_path).joinpath(ss)
        images_output_path = dest.joinpath("images")
        annotations_output_path = dest.joinpath("annotations")
        if os.path.exists(images_output_path) and os.path.isdir(images_output_path):
            shutil.rmtree(images_output_path)
        if os.path.exists(annotations_output_path) and os.path.isdir(annotations_output_path):
            shutil.rmtree(annotations_output_path)
        #create output path
        if not os.path.exists(images_output_path):
            os.makedirs(images_output_path)
        if not os.path.exists(annotations_output_path):
            os.makedirs(annotations_output_path)
    train_img = Path(train_img)
    test_img = Path(test_img)
    
    for ss in subset:
        dest = Path(des_path).joinpath(ss)
        images_output_path = dest.joinpath("images")
        annotations_output_path = dest.joinpath("annotations")
        
        img_list = train_img.iterdir() if ss=="train" else test_img.iterdir()
        gt_path = train_gt if ss=="train" else test_gt
        gt_path = Path(gt_path)

        for file_name in img_list:
            img_name = str(file_name)
            img_name = img_name[img_name.rfind('img'):]
            gt_name = "gt_" + img_name[:-4] + ".txt"
            ann_path = gt_path.joinpath(gt_name)
            try:
                lines = open(str(ann_path), encoding='utf-8').readlines()
            except:
                continue
            try:
                img = cv2.imread(str(file_name))
            except:
                continue
            data = dict()
            data["file"] = img_name
            data["width"] = img.shape[1]
            data["height"] = img.shape[0]
            data["depth"] = img.shape[2]
            data["words"] = list()

            for line in lines:
                text = dict()
                ori_box = line.strip().encode('utf-8').decode('utf-8-sig').split(',')
                box = [int(ori_box[j]) for j in range(8)]
                word = ori_box[8:]
                word = word[0].strip().strip('"\"')
                print(word)
                text["text"] = word
                text["type"] = "None"
                text["x1"] = box[0]
                text["y1"] = box[1]
                text["x2"] = box[2]
                text["y2"] = box[3]
                text["x3"] = box[4]
                text["y3"] = box[5]
                text["x4"] = box[6]
                text["y4"] = box[7]
                text["chars"] = []
                data["words"].append(text)

            copy(file_name, images_output_path)
            with open(os.path.join(annotations_output_path, data["file"][:-3] + "json"), "w") as file:
                    json.dump(data, file, indent=4)
            count+=1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_img', required=True, help='path to train image')
    parser.add_argument('--train_gt', required=True, help='path to train gt')
    parser.add_argument('--test_img', required=True, help='path to test image')
    parser.add_argument('--test_gt', required=True, help='path to test gt')
    parser.add_argument('--des_path', required=True, help='path to destination output path (contain images and annotations subfolder)')

    opt = parser.parse_args()
    convert(train_img=opt.train_img, train_gt=opt.train_gt, test_img=opt.test_img, test_gt= opt.test_gt, des_path=opt.des_path)

if __name__ == '__main__':
    main()