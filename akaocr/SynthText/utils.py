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

def get_args():

    parser = argparse.ArgumentParser(description='Run SynthText')

    parser.add_argument('--backgrounds_path',
                        type = str, 
                        default = '.', 
                        help = 'The path of background directory, contains all background.')

    parser.add_argument('--vocab_path',
                        type = str, 
                        default = '.', 
                        help = 'The path of vocab files')

    parser.add_argument('--segment_path',
                        type = str,
                        default = '.', 
                        help = "The path to segmentation files (h5) generate with matlab. With an image didn't segment, it will be auto segment in another algorithism")

    #######################################

    parser.add_argument('--num_samples',
                        type = int, 
                        default = 100, 
                        help = 'The number of out images for each backgroud image')
                        
    parser.add_argument('--aug_percent',
                        type = float, 
                        default = 0.5, 
                        help = 'The percent of augumentation of each box')

    parser.add_argument('--max_num_box',
                        type = int, 
                        default = 100, 
                        help = 'The maximum number of box will generators for each image')

    parser.add_argument('--box_iter',
                        type = int,
                        default = 100, 
                        help = 'The maximum tries when gen a box')

    parser.add_argument('--fixed_size',
                        type = tuple, 
                        default = None)

    parser.add_argument('--weigh_random_range',
                        type = tuple, 
                        default = None, 
                        help = 'Tuple with range of weigh')

    parser.add_argument('--heigh_random_range',
                        type = tuple, 
                        default = None, 
                        help = 'Tuple with range of heigh')

    #######################################

    parser.add_argument('--fonts_path',
                        type = str, 
                        default = '.', 
                        help = 'The path of font directory, just for True Tye Font (.ttf) files.')

    parser.add_argument('--source_text_path',
                        type = str, 
                        default = '.', 
                        help = 'The path of source text path')

    parser.add_argument('--random_color',
                        type = boolean,
                        default = False,  
                        help = 'Boolean')

    parser.add_argument('--font_color',
                        type = list,
                        default = [0,0,0],  
                        help = 'Font color')

    parser.add_argument('--output_path',
                        type = str, 
                        default = '.', 
                        help = 'The path to save output images and anotations')

    parser.add_argument('--method',
                        type = str,
                        default = 'blacklist',  
                        help = 'Method')


    return parser.parse_args()                                  