import os
import cv2
import numpy as np
import imgaug.augmenters as iaa
import json
from PIL import Image
import copy
from pathlib import Path
from pipeline.util import Visualizer

class Augmentation():
    def __init__(self, cfg, option):
        self.cfg = cfg
        self.option = option
        
    def augment(self, images, points=None, imwrite=False, output_path=None):
        """
        Augmentation list of images.
        Option list: Dictionary of name of augmentation: {augment name: augment value}
        @param images: list of images
        @param points: List of info json
        @type option: dict
        Example:
        option = {'shear'  :{'p':0.8,'v':{"x": (-15, 15), "y": (-15, 15)}},
                'dropout':{'p':0.6,'v':(0.2,0.3)},
                'blur'   :{'p':0.6,'v':(0.0, 2.0)}}
        """
        labels = copy.deepcopy(points)
        if self.option is None:
            return images, points
        base = iaa.Sequential()
        if 'shear' in self.option:
            base.add(iaa.Sometimes(self.option['shear']['p'],
                                iaa.Affine(shear=self.option['shear']['v'],
                                            fit_output=True,
                                            backend='cv2')))
        if 'dropout' in self.option:
            base.add(iaa.Sometimes(self.option['dropout']['p'],
                                iaa.Dropout(p=self.option['dropout']['v'])))
        if 'blur' in self.option:
            base.add(iaa.Sometimes(self.option['blur']['p'],
                                iaa.GaussianBlur(sigma=self.option['blur']['v'])))

        # base.add(iaa.Sometimes(0.85,
        #                        iaa.ElasticTransformation(alpha=(40, 60), sigma=(6, 12))))
        if 'elastic' in self.option:
            base.add(iaa.Sometimes(self.option['elastic']['p'],
                                iaa.ElasticTransformation(alpha=(40,60), sigma=(6,12))))
                                
        if points is not None:
            kps = []
            box_origin = []
            for img_info in points:
                kp = []
                for char_inf in img_info['words']:
                    ck = [(char_inf['x1'], char_inf['y1']),
                        (char_inf['x2'], char_inf['y2']),
                        (char_inf['x3'], char_inf['y3']),
                        (char_inf['x4'], char_inf['y4'])]
                    kp.extend(ck)
                    box_origin.append([[char_inf['x1'], char_inf['y1']], [char_inf['x2'], char_inf['y2']],
                                        [char_inf['x3'], char_inf['y3']], [char_inf['x4'], char_inf['y4']]])
                    for char in char_inf['chars']:
                        ck = [(char['x1'], char['y1']),
                            (char['x2'], char['y2']),
                            (char['x3'], char['y3']),
                            (char['x4'], char['y4'])]
                        kp.extend(ck)
                        box_origin.append([[char['x1'], char['y1']], [char['x2'], char['y2']],
                                        [char['x3'], char['y3']], [char['x4'], char['y4']]])
                kps.append(kp)
            
            images_aug, out_keypoints = base(images=[255 - np.array(img) for img in images], keypoints=kps)
            for i in range(len(points)):
                z = 0
                box_augment = list()
                for j in range(len(points[i]['words'])):
                    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = out_keypoints[i][4 * (j+z):4 * (j+z+1)]
                    points[i]['words'][j]['x1'] = int(x1)
                    points[i]['words'][j]['x2'] = int(x2)
                    points[i]['words'][j]['x3'] = int(x3)
                    points[i]['words'][j]['x4'] = int(x4)
                    points[i]['words'][j]['y1'] = int(y1)
                    points[i]['words'][j]['y2'] = int(y2)
                    points[i]['words'][j]['y3'] = int(y3)
                    points[i]['words'][j]['y4'] = int(y4)
                    box_augment.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
                    for k in range(len(points[i]['words'][j]['chars'])):
                        (x1, y1), (x2, y2), (x3, y3), (x4, y4) = out_keypoints[i][4 * (j+z+1):4 * (j+z+2)]
                        points[i]['words'][j]['chars'][k]['x1'] = int(x1)
                        points[i]['words'][j]['chars'][k]['x2'] = int(x2)
                        points[i]['words'][j]['chars'][k]['x3'] = int(x3)
                        points[i]['words'][j]['chars'][k]['x4'] = int(x4)
                        points[i]['words'][j]['chars'][k]['y1'] = int(y1)
                        points[i]['words'][j]['chars'][k]['y2'] = int(y2)
                        points[i]['words'][j]['chars'][k]['y3'] = int(y3)
                        points[i]['words'][j]['chars'][k]['y4'] = int(y4)
                        box_augment.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
                        z +=1
    
        else:
            images_aug = base(images=[255 - np.array(img) for img in images])
        imgs = [255 - np.array(img) for img in images_aug]
    
        if imwrite and output_path:
            # data_path = "../data/visualize_augment/"
            # Path(data_path).mkdir(parents=True, exist_ok=True)
            # name = os.path.join(data_path, points[0]['file'])
            # vis = Visualizer(output_folder = data_path)
            # img = vis.visualizer(image_ori=np.array(images[0]), contours=np.array(box_origin), show=False)
            # cv2.imwrite(name, img)
            # name = os.path.join(data_path, "augmented" + points[0]['file'])
            # img = vis.visualizer(image_ori=np.array(imgs[0]), contours=np.array(box_augment), show=False)
            # cv2.imwrite(name, img)

            name = points[0]['file']
            self.imwrite(output_path, name, np.array(images[0]), np.array(box_origin))

            name = "augmented" + points[0]['file']
            self.imwrite(output_path, name, np.array(imgs[0]), np.array(box_augment))
        
        return  imgs[0], points[0]

    def imwrite(self, output_path, name, img, contour):
        Path(output_path).mkdir(parents=True, exist_ok=True)
        name = os.path.join(output_path, name)
        vis = Visualizer(output_folder = output_path)
        img = vis.visualizer(image_ori=img, contours=contour, show=False)
        cv2.imwrite(name, img)