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

def convert(train_img, train_gt, train_ch_gt, test_img, test_gt, des_path):
    #remove if exist
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
        ch_gt_path = Path(train_ch_gt)

        for file_name in img_list:
            img_name = str(file_name)
            img_name = img_name[img_name.rfind('img'):] if ss=="test" else img_name[img_name.rfind('.jpg')-3:]
            gt_name = "gt_" + img_name[:-4] + ".txt"
            ch_gt_name = img_name[:-4] + "_GT.txt"
            ann_path = gt_path.joinpath(gt_name)
            ch_ann_path = ch_gt_path.joinpath(ch_gt_name)
            ch_lines = []
            try:
                lines = open(str(ann_path), encoding='utf-8').readlines()
                ch_lines = open(str(ch_ann_path), encoding='utf-8').readlines()
            except:
                continue

            word_line = dict()
            w_count = 1
            ch_count = 1
            for index in range(len(ch_lines)):
                if ch_lines[index] != '\n' and ch_lines[index][0]!='#':
                    word_line[str(w_count)+'_'+str(ch_count)] = ch_lines[index]
                    ch_count += 1
                elif ch_lines[index]=='\n':
                    w_count += 1
                    ch_count = 1
                elif ch_lines[index][0]=='#' and ch_lines[index-1]=='\n' or index==0 and ch_lines[index][0]=='#':
                    w_count -= 1
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
            current_word = 0
            for line in lines:
                current_word+=1
                text = dict()
                ori_box = line.strip().encode('utf-8').decode('utf-8-sig').split(' ') if ss=="train" else line.strip().encode('utf-8').decode('utf-8-sig').split(', ')
                box = [int(ori_box[j]) for j in range(4)]
                word = ori_box[4:]
                word = word[0].strip().strip('"\"')
                text["text"] = word
                text["type"] = "None"
                text["x1"] = box[0]
                text["y1"] = box[1]
                text["x2"] = box[2]
                text["y2"] = box[1]
                text["x3"] = box[2]
                text["y3"] = box[3]
                text["x4"] = box[0]
                text["y4"] = box[3]
                text["chars"] = []
                if ss=="train":
                    len_word = len(word)
                    for i in range(1,len_word+1):
                        char = dict()
                        ch = word_line[str(current_word)+'_'+str(i)].split(' ')
                        ch_text = ch[9].strip().strip('"\"')
                        char["text"] = ch_text
                        char["x1"] = int(ch[5])
                        char["y1"] = int(ch[6])
                        char["x2"] = int(ch[7])
                        char["y2"] = int(ch[6])
                        char["x3"] = int(ch[7])
                        char["y3"] = int(ch[8])
                        char["x4"] = int(ch[5])
                        char["y4"] = int(ch[8])
                        text["chars"].append(char)
                data["words"].append(text)

            copy(file_name, images_output_path)
            with open(os.path.join(annotations_output_path, data["file"][:-3] + "json"), "w") as file:
                    json.dump(data, file, indent=4)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_img', required=True, help='path to train image')
    parser.add_argument('--train_gt', required=True, help='path to train gt')
    parser.add_argument('--train_ch_gt', required=True, help='path to train gt')
    parser.add_argument('--test_img', required=True, help='path to test image')
    parser.add_argument('--test_gt', required=True, help='path to test gt')
    parser.add_argument('--des_path', required=True, help='path to destination output path (contain images and annotations subfolder)')

    opt = parser.parse_args()
    convert(train_img=opt.train_img, train_gt=opt.train_gt, test_img=opt.test_img, train_ch_gt=opt.train_ch_gt, test_gt= opt.test_gt, des_path=opt.des_path)
    print("Convert icdar13 dataset done")
if __name__ == '__main__':
    main()