import cv2
import json
import numpy as np

class Transform():

    def __init__(self, source_images, source_chars_coor, target_points, target_image, ):
        
        assert len(source_images) == len(target_points)
        assert len(source_images) == len(source_chars_coor)

        self.source_images      = source_images
        self.source_chars_coor  = source_chars_coor
        self.target_points      = target_points

        if type(target_image)=="str":
            self.target_image   = cv2.imread(target_image)
        else:
            self.target_image   = target_image

        self.out_size           = (self.target_image.shape[1],self.target_image.shape[0])

    def transform(self,image,trans_matrix):

        return  cv2.warpPerspective(image,
                                    trans_matrix, 
                                    self.out_size,
                                    borderValue  = (255,255,255))

    def tranform_matrix(self,source_coor, target_coor):
        return  cv2.getPerspectiveTransform(source_coor,target_coor)

    def new_coordinate(self, p, matrix):
        px = (matrix[0][0]*p[0] + matrix[0][1]*p[1] + matrix[0][2]) / ((matrix[2][0]*p[0] + matrix[2][1]*p[1] + matrix[2][2]))
        py = (matrix[1][0]*p[0] + matrix[1][1]*p[1] + matrix[1][2]) / ((matrix[2][0]*p[0] + matrix[2][1]*p[1] + matrix[2][2]))
        return px,py

    def fit(self):
        
        result_pic = self.target_image.copy()
        out_json = {
                    "file": "sample.jpg",
                    "width" : self.out_size[0],
                    "height": self.out_size[1],
                    "depth": 3,
                    "words": []}
        for img,char_coor,target_point in zip(self.source_images,self.source_chars_coor, self.target_points):
            h,w,_ = np.asarray(img).shape
            source_point = np.float32([[(0,0),(w,0),(w,h),(0,h)]])
            matrix = self.tranform_matrix(source_point,target_point)
            word_img = self.transform(img, matrix)
            result_pic &= word_img
            word_out = {}
            word_out['text'] = char_coor['words']
            word_out['x1'] = target_point[0][0][0]
            word_out['y1'] = target_point[0][0][1]
            word_out['x2'] = target_point[0][1][0]
            word_out['y2'] = target_point[0][1][1]
            word_out['x3'] = target_point[0][2][0]
            word_out['y3'] = target_point[0][2][1]
            word_out['x4'] = target_point[0][3][0]
            word_out['y4'] = target_point[0][3][1]
            word_out['chars'] = []

            for char_dict in char_coor['text']:
                out_char = {}
                out_char['text'] = char_dict['char']
                out_char['x1'], out_char['y1'] = self.new_coordinate((char_dict['x1'],char_dict['y1']),matrix)
                out_char['x2'], out_char['y2'] = self.new_coordinate((char_dict['x2'],char_dict['y2']),matrix)
                out_char['x3'], out_char['y3'] = self.new_coordinate((char_dict['x3'],char_dict['y3']),matrix)
                out_char['x4'], out_char['y4'] = self.new_coordinate((char_dict['x4'],char_dict['y4']),matrix)
                word_out['chars'].append(out_char)

            out_json['words'].append(word_out)

            
        return result_pic, out_json
