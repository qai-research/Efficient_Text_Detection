import math
import os
from utils.data.collates import NormalizePAD, ResizeNormalize
from PIL import ImageFont, ImageDraw, Image
from pathlib import Path
import numpy as np
import cv2

import uuid

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

class Visualizer:
    """
    Utility class for visualizing image.

    Attributes
    ----------
    output_folder : str
        the path to output folder
    pre : str
        the prefix to append to out image's name
    suf : str
        the suffix to append to out image's name

    Methods
    -------
    imwrite(image, file_name)
        write the image to output_folder
    """

    def __init__(self, data_path='./data', output_folder='./', pre='', suf='random'):
        """
        Parameters
        ----------
        output_folder : str
            the path to output folder
        pre : str, optional, default: ""
            the prefix to be appended before the given image's name
        suf : str or "random", optional, default: "random"
            the suffix to be appended after the given image's name
        """
        self.data_path = Path(data_path)
        self.output_folder = Path(output_folder)
        self.pre = pre
        self.suf = suf

    def imwrite(self, image, file_name):
        """
        Parameters
        ----------
        image : numpy array
            the image to be written to file
        file_name : str
            the image's file name
        """
        file_path = Path(file_name)
        name_base = file_path.stem

        if self.suf == 'random':
            post = str(uuid.uuid4())[:8]
        else:
            post = self.suf

        new_name_base = self.pre + name_base + post
        file_type = file_path.suffix
        write_file_path = self.output_folder.joinpath(new_name_base + file_type)

        if not self.output_folder.exists():
            self.output_folder.mkdir()

        is_success, im_buf_arr = cv2.imencode(file_type, image)
        im_buf_arr.tofile(str(write_file_path))

    @staticmethod
    def draw_zone(image, zone, color, thickness):
        points = np.array([zone.points[0].to_array(),
                           zone.points[1].to_array(),
                           zone.points[2].to_array(),
                           zone.points[3].to_array()])
        points = points.reshape((-1, 1, 2))

        image = cv2.polylines(image, [points], isClosed=True, color=color, thickness=thickness)
        return image

    @staticmethod
    def draw_text(self, image, text, x, y, font_size=18, color=(0, 0, 0), font=None, thickness=3):
        img = np.copy(image)
        default_font = 'default_vis_font.ttf'
        #find font in data input folder
        if self.data_path!=os.path.join(self.data_path, default_font):
            font_lst = sorted(self.data_path.glob('*.ttf'))
            if len(font_lst) < 1:
                # font = None
                font = os.path.join(self.data_path, default_font)
            else:
                font = str(font_lst[0])
        
        if font is None:
            (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_size,
                                                        thickness=thickness)[0]
            text_offset_x = x
            text_offset_y = y

            # make the coords of the box with a small padding of two pixels
            box_coords = ((text_offset_x, text_offset_y),
                          (text_offset_x + text_width + 2, text_offset_y - text_height - 2))

            cv2.rectangle(img, box_coords[0], box_coords[1], (255, 255, 255), cv2.FILLED)
            cv2.putText(img, text, (text_offset_x, text_offset_y), cv2.FONT_HERSHEY_SIMPLEX,
                        font_size, color, thickness)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            im_pil = Image.fromarray(img)
            draw = ImageDraw.Draw(im_pil)
            u_font = ImageFont.truetype(font, font_size)
            draw.text((x, y), text, font=u_font, fill=color)
            img = cv2.cvtColor(np.array(im_pil), cv2.COLOR_RGB2BGR)
        return img

    def visualizer(self, image_ori, contours=None, boxes=None, lines=None, bcolor=(0, 255, 0), texts=None,
                   font=None, font_size=30, thick=2, windows=None, show=False, name='demo', tcolor=(255, 0, 0), 
                   gt_text=None, gt_color=(0, 0, 255)):
        image = image_ori.copy()
        imshape = image.shape
        
        if len(imshape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        if windows is None:
            windows = imshape[:2]

        if contours is not None:
            image = cv2.drawContours(image, contours.astype(int), -1, bcolor, thick)
            if texts is not None:
                if gt_text is None:
                    for con, tex in zip(contours, texts):
                        image = self.draw_text(self, image, tex, con[3][0], con[3][1], font=font, font_size=font_size,
                                               color=tcolor)
                else:
                    for con, tex, gt in zip(contours, texts, gt_text):
                        if tex == gt:
                            image = self.draw_text(self, image, tex, con[3][0], con[3][1], font=font, font_size=font_size,
                                                   color=gt_color)
                        else:
                            image = self.draw_text(self, image, tex, con[3][0], con[3][1], font=font, font_size=font_size,
                                                   color=tcolor)

        elif boxes is not None:
            for b in boxes:
                image = cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), bcolor, thick)
            if texts is not None:
                if gt_text is None:
                    for box, tex in zip(boxes, texts):
                        image = self.draw_text(self, image, tex, box[0], box[3], font=font, font_size=font_size, color=tcolor)
                else:
                    for box, tex, gt in zip(boxes, texts, gt_text):
                        if tex == gt:
                            image = self.draw_text(self, image, tex, box[0], box[3], font=font, font_size=font_size,
                                                   color=gt_color)
                        else:
                            image = self.draw_text(self, image, tex, box[0], box[3], font=font, font_size=font_size,
                                                   color=tcolor)

        if lines is not None:
            for li in lines:
                li = li[0]
                image = cv2.line(image, (li[0], li[1]), (li[2], li[3]), (255, 0, 0), thick, cv2.LINE_AA)

        if show:
            cv2.namedWindow(name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(name, windows[0], windows[1])
            cv2.imshow(name, image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return image

def experiment_loader(name='test', type='detec', model_format='pth', data_path='./data'):
    data_path = Path(data_path)
    if type == 'detec':
        saved_models_path = 'exp_detec'
    else:
        saved_models_path = 'exp_recog'
    data_path = data_path.joinpath(saved_models_path, name)
    if not data_path.exists():
        raise Exception("No experiment folder for", name)
    if model_format=='pth':
        saved_model = sorted(data_path.glob('*.pth'))
    elif model_format=='onnx':
        saved_model = sorted(data_path.glob('*.onnx'))
    elif model_format=='pt':
        saved_model = sorted(data_path.glob('*.pt'))
    elif model_format=='trt':
        saved_model = sorted(data_path.glob('*.trt'))
    saved_config = sorted(data_path.glob('*.yaml'))

    if len(saved_model) < 1:
        raise Exception("No model for experiment ", name, type, "in", data_path)
    if len(saved_config) < 1:
        raise Exception("No config for experiment ", name, type, "in", data_path)

    return str(saved_model[0]), str(saved_config[0])