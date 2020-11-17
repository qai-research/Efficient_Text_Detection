import random
import numpy as np
from PIL import Image

from utils.file_utils import read_json_annotation
from pre.image import ImageProc
from utils.visproc import save_heatmap


def load_image_boxes(image, json_label, min_size=768, max_size=1280):
    """
    Load bounding boxes and covert image + label to random size between min_size and max_side
    :param image: numpy array
    :param json_label: json label
    :param min_size: min size of random image
    :param max_size: max size of random image
    :return: np.array image, list polygons, list words
    """
    words, chars_ann = read_json_annotation(json_label)
    image, scale = ImageProc.random_scale(image, min_size, max_size)
    confidences = []
    character_boxes = []
    for i, cha in enumerate(chars_ann):
        bboxes = []
        for j, bo in enumerate(cha):
            bboxes.append(np.array(bo * scale))
        character_boxes.append(np.array(bboxes))
    return image, character_boxes, words


def transform2heatmap(image, json_label, gaussian_transformer, min_size=768, max_size=1280):
    """
    Convert data to heatmap representation for training
    :param image: numpy array
    :param json_label: json label
    :param gaussian_transformer: gaussian transformer for boxes
    :param min_size: min image size
    :param max_size: max image size
    :return: numpy image, text region matrix, affinity matrix, confident score
    """
    # gaussian_transformer = GaussianTransformer(img_size=512, region_threshold=0.35, affinity_threshold=0.15)

    image, character_bboxes, words = load_image_boxes(image, json_label, min_size=min_size, max_size=max_size)
    confidences = 1.0
    confidence_mask = np.ones((image.shape[0], image.shape[1]))

    region_scores = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
    affinity_scores = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)

    if len(character_bboxes) > 0:
        region_scores = gaussian_transformer.generate_region(region_scores.shape, character_bboxes)
        affinity_scores, affinity_bboxes = gaussian_transformer.generate_affinity(region_scores.shape,
                                                                                  character_bboxes,
                                                                                  words)
    random_transforms = [image, region_scores, affinity_scores, confidence_mask]
    random_transforms = ImageProc.sub_image_random_crop(random_transforms, (min_size, min_size), character_bboxes)

    cv_image, region_scores, affinity_scores, confidence_mask = random_transforms

    region_scores = ImageProc.resize_gt(region_scores, min_size//2)
    affinity_scores = ImageProc.resize_gt(affinity_scores, min_size//2)
    confidence_mask = ImageProc.resize_gt(confidence_mask, min_size//2)

    # image = Image.fromarray(cv_image)
    # image = image.convert('RGB')
    # image = transforms.ColorJitter(brightness=32.0 / 255, saturation=0.5)(image)
    # import cv2
    # primg = cv2.resize(cv_image, (350, 350))
    # save_heatmap(primg, [], region_scores, affinity_scores)

    image = ImageProc.normalize_mean_variance(np.array(cv_image), mean=(0.485, 0.456, 0.406),
                                              variance=(0.229, 0.224, 0.225))
    return image, region_scores, affinity_scores, confidence_mask, confidences






