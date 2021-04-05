import math
from utils.data.collates import NormalizePAD, ResizeNormalize
from PIL import ImageFont, ImageDraw, Image
from pathlib import Path
from utils.utility import initial_logger
logger = initial_logger()
import numpy as np
import cv2

class AlignCollate(object):
    def __init__(self, img_h=32, img_w=128, keep_ratio_with_pad=False):
        self.img_h = img_h
        self.img_w = img_w
        self.keep_ratio_with_pad = keep_ratio_with_pad

    def __call__(self, image):
        if self.keep_ratio_with_pad:  # same concept with 'Rosetta' paper
            resized_max_w = self.img_w
            input_channel = 3 if image.mode == 'RGB' else 1
            transform = NormalizePAD((input_channel, self.img_h, resized_max_w))

            w, h = image.size
            ratio = w / float(h)
            if math.ceil(self.img_h * ratio) > self.img_w:
                resized_w = self.img_w
            else:
                resized_w = math.ceil(self.img_h * ratio)

            resized_image = transform(image.resize((resized_w, self.img_h), Image.BICUBIC))
            image_tensors = resized_image.unsqueeze(0)

        else:
            transform = ResizeNormalize((self.img_w, self.img_h))
            image_tensor = transform(image)
            image_tensors = image_tensor.unsqueeze(0)

        return image_tensors  # (1, c, h, w)

def imagewriter(img, boxes, text_list, score_list, output_path, fontpath):
    font = ImageFont.truetype(font=fontpath, size=40)
    b,g,r = 0,0,255
    for i in range(len(boxes)):
        box = boxes[i]
        poly = np.array(box).astype(np.int32)
        if len(poly.shape) == 1:
            x0, y0 = poly[0], poly[1]
            x1, y1 = poly[2], poly[3]
        else:
            x0, y0 = np.min(poly, axis=0)
            x1, y1 = np.max(poly, axis=0)
        color = (0, 0, 255)
        thickness = 2
        img = cv2.rectangle(img, (x0,y0), (x1,y1), color=color, thickness=thickness)
        # img = cv2.putText(img, text_list[i], (x1, y1-60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color=color, thickness=thickness)
        img = cv2.putText(img, str(score_list[i].cpu().numpy())[:4], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color=color, thickness=thickness)
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        draw.text((x1, y1-80), text_list[i], font = font, fill = (b, g, r))
        img = np.array(img_pil)
    cv2.imwrite(output_path, img)

def experiment_loader(name='best_accuracy.pth', type='detec', data_path="../"):
    data_path = Path(data_path)
    if type == 'detec':
        saved_model_path = 'data/exp_detec/test'
    elif type == 'recog':
        saved_model_path = 'data/exp_recog/test'
    saved_model = data_path.joinpath(saved_model_path, name)
    if not saved_model.exists():
        logger.warning(f"No saved model name {name} in {saved_model_path}")
        logger.warning(f"Load latest saved model")
        saved_model_list = sorted(data_path.joinpath(saved_model_path).glob('*.pth'))
        if len(saved_model_list)<1:
            raise Exception("No model for experiment ", name, " in ", data_path.joinpath(saved_model_path))
        saved_model = str(saved_model_list[-1])
    return saved_model