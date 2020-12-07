import cv2
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa


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


def augmentation(images, points, option):
    """
    Augmentation list of images.
    Option list: Dictionary of name of augmentation: {augment name: augment value}
    @type option: dictionary
    Example:
    option = {'shear'  :{'p':0.8,'v':{"x": (-15, 15), "y": (-15, 15)}},
              'dropout':{'p':0.6,'v':(0.2,0.3)},
              'blur'   :{'p':0.6,'v':(0.0, 2.0)}}
    """

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
    images_aug, out_keypoints = base(images=255 - np.array(images), keypoints=kps)
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
    return 255 - np.array(images_aug), points
