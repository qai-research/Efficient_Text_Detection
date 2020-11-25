import os
import sys
import cv2
import pdb
import h5py
import copy
import math
import time
import random
import datetime
import operator
import numpy as np
import itertools as IT
import multiprocessing
from scipy import optimize
from functools import reduce
from matplotlib import pyplot as plt

sys.path.append('../akaocr')
from SynthText.utils import resize

class BoxGenerator():

    def __init__(self, img_path, fixed_size = None, weigh_random_range = None, 
                 heigh_random_range = None, box_iter = None, aug_percent = 0, 
                 num_samples = 100, max_num_box = 100, segment = None, threshold = 0.01):
        
#         if (weigh_random_range is None or heigh_random_range is None):
#             raise ValueError("weigh_random_range (tuple) and heigh_random_range (tuple) are required.")
        self.image = cv2.imread(img_path)
        self.fixed_size = fixed_size
        self.weigh_random_range = weigh_random_range
        self.heigh_random_range = heigh_random_range
        if self.fixed_size is None:
            self.min_box_w, self.max_box_w = self.weigh_random_range
            self.min_box_h, self.max_box_h = self.heigh_random_range
        else:
            self.min_box_w, self.min_box_h = self.fixed_size

        self.threshold = threshold
        self.min_box_size =  self.min_box_w * self.min_box_h
        self.box_iter = box_iter
        self.num_samples = num_samples
        self.max_num_box = max_num_box
        self.aug_percent = aug_percent
        self.target_name        = os.path.basename(img_path)
        self.target_base_name   = os.path.splitext(self.target_name)[0]
        self.target_base_type   = os.path.splitext(self.target_name)[1][1:]
        self.out_size           = (self.image.shape[1],self.image.shape[0])
        try:            
            if type(segment)==str:  
                f = h5py.File(segment, 'r')
                assert self.target_name in f['mask'].keys()
            else:
                f = segment.copy()
            val = np.array(f['mask'][self.target_name])
            m = np.max(val)
            self.segments = []
            for j in range(m):
                num = sum(sum(val==j+1))
                if num >2000:     
                    tem = copy.deepcopy(val)
                    tem[tem!=j] = 0   
                    self.segments.append(tem)
            self.box_gen = self.box_generator_existed_masker
        except Exception as e:
            self.box_gen = self.box_generator_with_masker_gen



    def mask_marker(self):
        dump_img = self.image.copy()
        b,g,r = cv2.split(dump_img)
        rgb_img = cv2.merge([r,g,b])

        gray = cv2.cvtColor(dump_img,cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)        
        kernel = np.ones((3,3),np.uint8)
        closing = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel, iterations = 3)
        sure_bg = cv2.dilate(closing,kernel,iterations=3)        
        dist_transform = cv2.distanceTransform(sure_bg,cv2.DIST_L2,3)
        ret, sure_fg = cv2.threshold(dist_transform,0.1*dist_transform.max(),255,0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg,sure_fg)
        ret, markers = cv2.connectedComponents(sure_fg)

        markers = markers + 1
        markers[unknown==255] = 0
        return cv2.watershed(dump_img,markers)

    def random_box(self, dump_marker, box_W, box_H):    
        all_valid_pixel = np.asarray(np.where(dump_marker[:,:,0]==255)).T
        y1,x1 = random.choice(all_valid_pixel)
        count = 500 if self.box_iter is None else self.box_iter  
        x2,y2 = x1+box_W , y1
        x3,y3 = x1+box_W , y1+box_H
        x4,y4 = x1       , y1+box_H
        box_coor = [(x1,y1),(x2,y2), (x3,y3),(x4,y4)]
        trans_box_coor = self.transform(box_coor)
        stop = False
        while True:
            y1,x1 = random.choice(all_valid_pixel)
            box_coor = [(x1,y1),(x1+box_W , y1), (x1+box_W , y1+box_H),(x1, y1+box_H)]
            trans_box_coor = self.transform(box_coor)
            count -= 1

            if count == 0:
                break
            if np.min(trans_box_coor)<0 or max([x for x,y in trans_box_coor])>self.out_size[0] or max([y for x,y in trans_box_coor])>self.out_size[1]:
                continue
            if np.all(dump_marker[self.get_inside(trans_box_coor)]!=0):
                break

        if count == 0:
            return None
        (x1,y1),(x2,y2), (x3,y3),(x4,y4) = trans_box_coor
        if x1==x3 or y1==y3:
            return None
        m = max(abs(y3-y1),abs(x3-x1))/min(abs(y3-y1),abs(x3-x1))
        x1 = np.random.randint(x1-m,x1+1)
        y1 = np.random.randint(y1-m,y1+1)
        x2 = np.random.randint(x2,x2+m+1)
        y2 = np.random.randint(y2-m,y2+1)
        x3 = np.random.randint(x3,x3+m+1)
        y3 = np.random.randint(y3,y3+m+1)
        x4 = np.random.randint(x4-m,x4+1)
        y4 = np.random.randint(y4,y4+m+1)
        return (x1,y1),(x2,y2), (x3,y3),(x4,y4)
    
    def get_inside(self, box_coor):
        empty_img = np.zeros_like(self.image)
        empty_img = cv2.fillPoly(empty_img, np.int32([box_coor]),[255,255,255])
        return np.where(empty_img==[255,255,255])
        


    def clockwise_sorted(self,coords):
        center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), coords), [len(coords)] * 2))
        return np.int32([sorted(coords, 
                                key=lambda coord: 
                                (-135 - math.degrees(math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360)])

    def get_box_size(self, dump_marker):        
        
        all_valid_pixel = np.asarray(np.where(dump_marker[:,:,0]==255)).T
        if self.fixed_size is None:
            max_W = max(self.weigh_random_range)
            max_H = max(self.heigh_random_range)
            if max_W*max_H >= len(all_valid_pixel):
                box_W = len(all_valid_pixel)/self.out_size[1]
                box_H = len(all_valid_pixel)/self.out_size[0]
            else:
                box_W = np.random.randint(*self.weigh_random_range)
                box_H = np.random.randint(*self.heigh_random_range)
        else:
            box_W, box_H = self.fixed_size

        return int(box_W), int(box_H)

    def box_generator_with_masker_gen(self):
        markers = self.mask_marker()
        img = self.image.copy()
        img[markers == 1] = [255,255,255]
        img[markers != 1] = [0,0,0]
        box_W, box_H = self.get_box_size(img)

        results = []
        plt.imshow(img)
        plt.show()
        for i in range(self.max_num_box):
            box_W, box_H = self.get_box_size(img)
            if box_W*box_H==0:
                continue
            p = self.random_box(img,box_W, box_H)
            if p is not None:
                img = cv2.fillPoly(img, np.int32([p]),[0,0,0])
                results.append(p)
        return results

    def box_generator_existed_masker(self):

        results = [] 
        a = time.time()
        for i,markers in enumerate(self.segments): 
            h,w = markers.shape
            dump_marker = np.zeros((h,w,3))
            dump_marker[markers==0] = [255,255,255]
            dump_marker = resize(dump_marker,self.out_size)
            num_box = 0
            img = self.image.copy()
            img[dump_marker != 255] = 255
            img[dump_marker == 255] = 0        
            grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 1, cv2.THRESH_BINARY)

            black_percent = np.sum(blackAndWhiteImage)/(blackAndWhiteImage.size)
            if black_percent<self.threshold:
                p = self.gen_fixed_box(blackAndWhiteImage)
                if p is not None:
                    img = cv2.fillPoly(img, np.int32([p]),[0,0,0])
                    results.append(self.normalize(p))
                    num_box+=1
            else:
                while True:
                    box_W, box_H = self.get_box_size(img)
                    if box_W*box_H==0:
                        continue
                    p = self.random_box(img,box_W, box_H)
                    if p is not None:
                        img = cv2.fillPoly(img, np.int32([p]),[0,0,0])
                        results.append(p)
                        num_box+=1
                    if random.random()<num_box/self.max_num_box:
                        break
                    if len(results)>self.max_num_box:
                        break
                    if random.random()>5*black_percent:
                        break

            if len(results)>self.max_num_box:
                break
        return results

    def normalize(self,xy):
        (x1,y1),(x2,y2),(x3,y3),(x4,y4) = xy
        if abs(x3-x1)<abs(y3-y1):
            return (x4,y4),(x1,y1),(x2,y2),(x3,y3)
        return (x1,y1),(x2,y2),(x3,y3),(x4,y4)

    def transform(self,box_coor):
        if np.random.random()<self.aug_percent:
            d = np.random.choice(range(-90,90))
            theta = math.radians(d)
            (x1,y1),(x2,y2),(x3,y3),(x4,y4) = self.rotate(box_coor, theta)
            if all(np.array([x1,x2,x3,x4])<self.out_size[1]) and all(np.array([x1,x2,x3,x4])>=0):
                if all(np.array([y1,y2,y3,y4])<self.out_size[0]) and all(np.array([y1,y2,y3,y4])>=0):
                    return (x1,y1),(x2,y2),(x3,y3),(x4,y4)
        return box_coor
            
        
    def rotate(self,xy, theta):
        
        def _rotate(xy,theta):
            cos_theta, sin_theta = math.cos(theta), math.sin(theta)
            return (
                xy[0] * cos_theta - xy[1] * sin_theta,
                xy[0] * sin_theta + xy[1] * cos_theta
            )


        def translate(xy, offset):
            return xy[0] + offset[0], xy[1] + offset[1]
        
        (x1,y1),(x2,y2),(x3,y3),(x4,y4) = xy
        w = abs(x3-x1)
        h = abs(y3-y1)
        offset = (x1,y1)
        new_p = [(0,0),(w,0),(w,h),(0,h)]
        return [translate(_rotate(xy, theta), offset) for xy in new_p]
        
            
        
    def gen_fixed_box(self, matrix, iters = 50,thresh = 0.5):
        all_valid_pixcel = np.asarray(np.where(matrix == 1)).T
        if len(all_valid_pixcel)==0:
            return None
        max_square = 0
        results = None
        for i in range(iters):
            c = random.choice(all_valid_pixcel)
            c_y, c_x = int(c[0]),int(c[1])
            m_x= min([x for (y,x) in all_valid_pixcel if y == c_y])
            m_y= min([y for (y,x) in all_valid_pixcel if x == c_x])
            M_x= max([x for (y,x) in all_valid_pixcel if y == c_y])
            M_y= max([y for (y,x) in all_valid_pixcel if x == c_x])
            if (M_y-M_x)*(m_y-m_x)/len(all_valid_pixcel)<0.3:
                continue
            while np.any(matrix[m_y:M_y+1,m_x:M_x+1]==0):
                if random.random()<0.5:
                    m_x+=1
                if random.random()<0.5:
                    M_x-=1
                if random.random()<0.5:
                    m_y+=1
                if random.random()<0.5:
                    M_y-=1
            s = np.sum(matrix[m_y:M_y+1,m_x:M_x+1])
            if s>max_square and s/matrix.size>thresh:
                d = np.random.choice(range(-2,2))
                theta = math.radians(d)
                results = self.rotate(((m_x,m_y),(M_x,m_y),(M_x,M_y),(m_x,M_y)),theta)
        return results

    def run(self):
        self.mask_marker()
        results = []
        for _ in range(self.num_samples):
            uniq_filename = str(datetime.datetime.now().date()) + '_' + str(datetime.datetime.now().time()).replace(':', '.')
            name = ".".join([self.target_base_name,uniq_filename,self.target_base_type])
            out_json = {"file": name,
                        "width" : self.out_size[0],
                        "height": self.out_size[1],
                        "depth": 3,
                        "words": []}
            for (x1,y1),(x2,y2),(x3,y3),(x4,y4) in self.box_gen():
                try:
                    word_len = max(abs((y3-y1)//(x3-x1)),abs((x3-x1)//(y3-y1)))
                    word_out = {}
                    word_out['text'] = 'x'*int(word_len)
                    word_out['x1'] = x1
                    word_out['y1'] = y1
                    word_out['x2'] = x2
                    word_out['y2'] = y2
                    word_out['x3'] = x3
                    word_out['y3'] = y3
                    word_out['x4'] = x4
                    word_out['y4'] = y4
                    word_out['chars'] = []
                    out_json['words'].append(word_out)
                except:
                    pass
            results.append(out_json)
                
        return results

