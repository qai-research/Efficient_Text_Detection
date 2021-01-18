import os
import cv2
import numpy as np
import imgaug as ia
import pandas as pd
import imgaug.augmenters as iaa
from .data_loader import lmdb_dataset_loader


def tranform_matrix(source_coor, target_coor):
    """
    Create transform matrix
    """
    return cv2.getPerspectiveTransform(source_coor, target_coor)


def coordinate_transform(p, source_coor, target_coor):
    """
    Coordinate transform
    """
    matrix = cv2.getPerspectiveTransform(source_coor, target_coor)
    px = (matrix[0][0] * p[0] + matrix[0][1] * p[1] + matrix[0][2]) / (
        (matrix[2][0] * p[0] + matrix[2][1] * p[1] + matrix[2][2]))
    py = (matrix[1][0] * p[0] + matrix[1][1] * p[1] + matrix[1][2]) / (
        (matrix[2][0] * p[0] + matrix[2][1] * p[1] + matrix[2][2]))
    return px, py


def resize_with_char(image, new_size):
    """
    resize with character
    """
    if len(image.shape) == 3:
        h, w, _ = image.shape
    else:
        h, w = image.shape
    n_w, n_h = new_size
    p1 = np.float32([[(0, 0), (w, 0), (w, h), (0, h)]])
    p2 = np.float32([[(0, 0), (n_w, 0), (n_w, n_h), (0, n_h)]])
    trans_matrix = cv2.getPerspectiveTransform(p1, p2)

    return cv2.warpPerspective(image,
                               trans_matrix,
                               new_size,
                               borderValue=(255, 255, 255))


def inpainting(image, poly_point):
    """
    Delete the object and change the color of the deleted area
    """
    clearned_target_image = image.copy()
    cv2.fillPoly(clearned_target_image, np.int32(poly_point), [255, 255, 255])
    mask_img = np.uint8(np.zeros(clearned_target_image.shape))
    cv2.fillPoly(mask_img, np.int32(poly_point), [255, 255, 255])
    mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    return cv2.inpaint(clearned_target_image, mask_img, 3, flags=cv2.INPAINT_NS)


def Augmentator(images, points=None, option=None):
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
    if option is None:
        return images, points
    base = iaa.Sequential()
    if 'shear' in option:
        base.add(iaa.Sometimes(option['shear']['p'],
                               iaa.Affine(shear=option['shear']['v'],
                                          fit_output=True,
                                          backend='cv2')))
    if 'dropout' in option:
        base.add(iaa.Sometimes(option['dropout']['p'],
                               iaa.Dropout(p=option['dropout']['v'])))
    if 'blur' in option:
        base.add(iaa.Sometimes(option['blur']['p'],
                               iaa.GaussianBlur(sigma=option['blur']['v'])))

    base.add(iaa.Sometimes(0.2,
                               iaa.ElasticTransformation(alpha=(40,60), sigma=(8,12))))
    if points is not None:
        kps = []
        for img_info in points:
            kp = []
            for char_inf in img_info['text']:
                ck = [(char_inf['x1'], char_inf['y1']),
                      (char_inf['x2'], char_inf['y2']),
                      (char_inf['x3'], char_inf['y3']),
                      (char_inf['x4'], char_inf['y4'])]
                kp.extend(ck)
            kps.append(kp)
        images_aug, out_keypoints = base(images=[255 - np.array(img) for img in images], keypoints=kps)
        for i in range(len(points)):
            for j in range(len(points[i]['text'])):
                (x1, y1), (x2, y2), (x3, y3), (x4, y4) = out_keypoints[i][4 * j:4 * (j + 1)]
                points[i]['text'][j]['x1'] = int(x1)
                points[i]['text'][j]['x2'] = int(x2)
                points[i]['text'][j]['x3'] = int(x3)
                points[i]['text'][j]['x4'] = int(x4)
                points[i]['text'][j]['y1'] = int(y1)
                points[i]['text'][j]['y2'] = int(y2)
                points[i]['text'][j]['y3'] = int(y3)
                points[i]['text'][j]['y4'] = int(y4)
    else:
        images_aug = base(images=[255 - np.array(img) for img in images])

    return [255 - np.array(img) for img in images_aug], points


def check_valid(dataframe, bg_df, source_df, fonts_df):
    """
    Check if the valid of input dataframe
    """
    df = dataframe.copy()
    results = {}
    for index, value in enumerate(dataframe.values):
        results[index] = {"Status": "valid", "Error": []}
        Method, NumCores, Fonts, Backgrounds, ObjectSources, Textsources, ImageSources = value[:7]
        if Backgrounds not in bg_df['NAME'].values:
            results[index]['Error'].append('Invalid Backgrounds Folder')
        else:
            info = bg_df[bg_df['NAME'] == Backgrounds]
            if Method == 'white' and 'white' not in info['METHOD'].values:
                results[index]['Error'].append('Invalid Method')
            if Fonts not in fonts_df['NAME'].values:
                results[index]['Error'].append('Fonts Folder Is Not Existed.')
            if str(Textsources) != '0' and Textsources not in source_df['NAME'].values:
                results[index]['Error'].append('The TextSources Is Not Existed.')
            if str(ObjectSources) != '0' and ObjectSources not in source_df['NAME'].values:
                results[index]['Error'].append('The ObjectSources Is Not Existed.')
            if str(ImageSources) != '0' and ImageSources not in source_df['NAME'].values:
                results[index]['Error'].append('The ObjectSources Is Not Existed.')
        if len(results[index]['Error']) is not 0:
            results[index]["Status"] = "INVALID"
    df['STATUS'] = [results[i]["Status"] for i in range(len(results))]
    df['DETAIL'] = [results[i]["Error"] for i in range(len(results))]
    return df


def get_all_valid(config):
    # CREATE BACKGROUND DATAFRAME
    existed_background = sorted(
        [os.path.join(config.background_folder, name) for name in os.listdir(config.background_folder)])
    whitelist_background = [path for path in existed_background if
                            os.path.isdir(path) and 'anotations' in os.listdir(path)]
    blacklist_background = [path for path in existed_background if os.path.isdir(path)]
    bg_df = {"NAME": [],
             "METHOD": [],
             "SIZE": [],
             "PATH": []
             }

    for path in existed_background:

        if not len(os.listdir(path)) > 0:
            continue

        if path in blacklist_background:
            bg_df['NAME'].append(os.path.basename(path))
            bg_df['METHOD'].append('black')
            bg_df['SIZE'].append(len(os.listdir(path + "/images")))
            bg_df['PATH'].append(path)

        if path in whitelist_background:
            bg_df['NAME'].append(os.path.basename(path))
            bg_df['METHOD'].append('white')
            bg_df['SIZE'].append(len(os.listdir(path + "/images")))
            bg_df['PATH'].append(path)
    bg_df = pd.DataFrame(bg_df, columns=["NAME", "METHOD", "SIZE", "PATH"])

    # CREATE SOURCE DATAFRAME
    existed_source = sorted([os.path.join(config.source_folder, name) for name in os.listdir(config.source_folder)])
    source_df = {"NAME": [],
                 "SIZE": [],
                 "PATH": [],
                 "TYPE": []
                 }
    for path in existed_source:
        if os.path.isfile(path) and path.endswith('.txt'):
            source_df["NAME"].append(os.path.basename(path))
            with open(path, 'r', encoding='utf8') as fr:
                source_df["SIZE"].append(len(fr.read().split("\n")))
            source_df["PATH"].append(path)
            source_df["TYPE"].append("Text")

        elif os.path.isdir(path) and 'images' in os.listdir(path):
            length = len(os.listdir(os.path.join(path, 'images')))
            source_df["NAME"].append(os.path.basename(path))
            source_df["SIZE"].append(length)
            source_df["PATH"].append(path)
            source_df["TYPE"].append("Object")

        elif os.path.isdir(path) and 'data.mdb' in os.listdir(path) and 'lock.mdb' in os.listdir(path):
            loader = lmdb_dataset_loader(path)
            length = len(loader.key_dict)
            source_df["NAME"].append(os.path.basename(path))
            source_df["SIZE"].append(length)
            source_df["PATH"].append(path)
            source_df["TYPE"].append("Handwriting Images")
    source_df = pd.DataFrame(source_df, columns=["NAME", "TYPE", "SIZE", "PATH"])

    # CREATE FONT DATAFRAME
    existed_font = sorted([os.path.join(config.font_folder, name) for name in os.listdir(config.font_folder)])

    font_df = {"NAME": [],
               "SIZE": [],
               "PATH": []
               }
    for path in existed_font:
        if os.path.isdir(path):
            font_df["NAME"].append(os.path.basename(path))
            font_df["SIZE"].append(len(os.listdir(path)))
            font_df["PATH"].append(path)
    font_df = pd.DataFrame(font_df, columns=["NAME", "SIZE", "PATH"])
    return bg_df, source_df, font_df
