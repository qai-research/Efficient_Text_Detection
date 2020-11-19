import cv2
import numpy as np

def tranform_matrix(source_coor, target_coor):
    return  cv2.getPerspectiveTransform(source_coor,target_coor)
    
def coordinate_transform(p, source_coor, target_coor):
    matrix = cv2.getPerspectiveTransform(source_coor,target_coor)
    px = (matrix[0][0]*p[0] + matrix[0][1]*p[1] + matrix[0][2]) / ((matrix[2][0]*p[0] + matrix[2][1]*p[1] + matrix[2][2]))
    py = (matrix[1][0]*p[0] + matrix[1][1]*p[1] + matrix[1][2]) / ((matrix[2][0]*p[0] + matrix[2][1]*p[1] + matrix[2][2]))
    return px,py

def resize(image, new_size):
    try:
        h,w,_ = image.shape
    except:
        h,w = image.shape
    n_w, n_h = new_size
    p1 = np.float32([[(0,0),(w,0),(w,h),(0,h)]])
    p2 = np.float32([[(0,0),(n_w,0),(n_w,n_h),(0,n_h)]])
    trans_matrix = cv2.getPerspectiveTransform(p1,p2)

    return  cv2.warpPerspective(image,
                                trans_matrix, 
                                new_size,
                                borderValue  = (255,255,255)) 