import cv2
import numpy as np


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
    elif len(image.shape) == 2:
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
