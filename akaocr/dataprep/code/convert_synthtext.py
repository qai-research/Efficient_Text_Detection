import torch
import scipy.io
import os
import json
from pathlib import Path
from shutil import copy
import cv2
import re
import itertools
import argparse

def convert(data_path, des_path):
    print('Converting groundtruth for synthtext800k...')
    print('There are n*100000 images, n are: ')
    data_path = Path(data_path)
    gt_path = data_path.joinpath("gt.mat")
    mat = scipy.io.loadmat(gt_path)

    des_path = Path(des_path)
    count = 0
    for i in range(len(mat["imnames"][0])):
        if i % 100000 == 0:
            count += 1
            print(count)
        sub_path = des_path.joinpath("synthtext_p"+str(count))
        save_img_path = des_path.joinpath(sub_path, "images")
        save_ann_path = des_path.joinpath(sub_path, "annotations")
        if not os.path.exists(save_img_path):
            os.makedirs(save_img_path)
        if not os.path.exists(save_ann_path):
            os.makedirs(save_ann_path)
        img_path = data_path.joinpath(mat["imnames"][0][i][0])
        data = dict()
        try:
            img = cv2.imread(str(img_path))
        except:
            continue
        data["file"] = mat["imnames"][0][i][0].split("/")[-1]
        data["width"] = img.shape[1]
        data["height"] = img.shape[0]
        data["depth"] = img.shape[2]
        data["words"] = list()
        word_list = list()

        word_list = [re.split(' \n|\n |\n| ', t.strip()) for t in mat["txt"][0][i]]
        word_list = list(itertools.chain(*word_list))
        word_list = [t for t in word_list if len(t) > 0]
        text_length = 0

        if len(word_list) == mat["wordBB"][0][i].shape[-1]:
            copy(img_path, save_img_path)
            for j in range(len(word_list)):
                text = dict()
                text["text"] = word_list[j]
                text["type"] = "None"
                text["x1"] = round((mat["wordBB"][0][i][0][0][j]).item())
                text["y1"] = round((mat["wordBB"][0][i][1][0][j]).item())
                text["x2"] = round((mat["wordBB"][0][i][0][1][j]).item())
                text["y2"] = round((mat["wordBB"][0][i][1][1][j]).item())
                text["x3"] = round((mat["wordBB"][0][i][0][2][j]).item())
                text["y3"] = round((mat["wordBB"][0][i][1][2][j]).item())
                text["x4"] = round((mat["wordBB"][0][i][0][3][j]).item())
                text["y4"] = round((mat["wordBB"][0][i][1][3][j]).item())
                text["chars"] = list()
                word = word_list[j]
                for k in range(len(word)):
                    char = dict()
                    char["text"] = word[k]
                    char["x1"] = round((mat["charBB"][0][i][0][0][text_length]).item())
                    char["y1"] = round((mat["charBB"][0][i][1][0][text_length]).item())
                    char["x2"] = round((mat["charBB"][0][i][0][1][text_length]).item())
                    char["y2"] = round((mat["charBB"][0][i][1][1][text_length]).item())
                    char["x3"] = round((mat["charBB"][0][i][0][2][text_length]).item())
                    char["y3"] = round((mat["charBB"][0][i][1][2][text_length]).item())
                    char["x4"] = round((mat["charBB"][0][i][0][3][text_length]).item())
                    char["y4"] = round((mat["charBB"][0][i][1][3][text_length]).item())
                    text["chars"].append(char)
                    text_length += 1
                data["words"].append(text)
        else:
            continue

        with open(os.path.join(save_ann_path, data["file"][:-3] + "json"), "w") as file:
                json.dump(data, file, indent=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True, help='path to source data (contain images and groundtruth .mat file')
    parser.add_argument('--des_path', required=True, help='path to destination output path (contain images and annotations subfolder)')

    opt = parser.parse_args()
    convert(data_path= opt.data_path, des_path=opt.des_path)

if __name__ == '__main__':
    main()
