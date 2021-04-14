import torch
import cv2
import numpy as np
import shutil

from models.detec.heatmap import HEAT
from models.detec.resnet_fpn_heatmap import HEAT_RESNET
from models.detec.efficient_heatmap import HEAT_EFFICIENT
from models.recog.atten import Atten
import os
from engine.config import setup, parse_base
from PIL import Image
import re
from models.modules.converters import AttnLabelConverter
import torch.nn.functional as F
from engine.infer.heat2boxes import Heat2boxes
from pipeline.util import AlignCollate, experiment_loader, Visualizer
from pre.image import ImageProc
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SlideWindow():
    """
    A basic slide window.
    Attributes
    ----------
    preprocess : callable
        the preprocess function whose input is a image,

    Methods
    -------
    __call__
        return list of windows from left to right, top to bottom
    """

    def __init__(self, window=(1280, 800), bufferx=50, buffery=20, preprocess=None):
        super().__init__()
        self.preprocess = preprocess
        self.window = window
        self.bufferx = bufferx
        self.buffery = buffery

    def __call__(self, img):
        if self.preprocess is not None:
            img = self.preprocess(img)

        original_shape = img.shape
        repeat_x = int(original_shape[1] // (self.window[0] - self.bufferx / 2) + 1)
        repeat_y = int(original_shape[0] // (self.window[1] - self.buffery / 2) + 1)
        # print(repeatx, repeaty)
        all_windows = []
        for i in range(repeat_y):
            perpen_window = []
            for j in range(repeat_x):
                crop_img = img[(self.window[1] - self.buffery) * i:(self.window[1] - self.buffery) * i + self.window[1],
                           (self.window[0] - self.bufferx) * j:(self.window[0] - self.bufferx) * j + self.window[0]]
                perpen_window.append(crop_img)
            all_windows.append(perpen_window)

        return all_windows


class Detectlayer():
    """
    A basic pipeline for performing detection.
    Attributes
    ----------
    preprocess : callable
        the preprocess function whose input is a image.
    detector : type of detector will be use (heatmap)

    config : path to config file for the detector, this will provide model with information and pretrained model
    Methods
    -------
    __call__
        execute the pipeline
    """

    def __init__(self, model_path=None, window=(1280, 800), bufferx=50, buffery=20):
        super().__init__()
        parse = parse_base()
        args = parse.parse_args()
        self.cfg = setup("detec", args)
        if model_path is None:
            model_path = experiment_loader(type='detec')
        if self.cfg.MODEL.NAME == "CRAFT":
            model = HEAT()
        elif self.cfg.MODEL.NAME == "RESNET":
            model = HEAT_RESNET()
        elif self.cfg.MODEL.NAME == "EFFICIENT":
            model = HEAT_EFFICIENT()
        model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
        model = model.to(device)
        self.window_shape = window
        self.bufferx = bufferx
        self.buffery = buffery
        self.detector = model
        self.cfg.MODEL.MAX_SIZE = 1200

    def detect(self, img):
        img, target_ratio = ImageProc.resize_aspect_ratio(
            img, self.cfg.MODEL.MAX_SIZE, interpolation=cv2.INTER_LINEAR
        )
        ratio_h = ratio_w = 1 / target_ratio
        img = ImageProc.normalize_mean_variance(img)
        img = torch.from_numpy(img).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
        img = (img.unsqueeze(0))  # [c, h, w] to [b, c, h, w]
        img = img.to(device)
        y,_ = self.detector(img)
        box_list = Heat2boxes(self.cfg, y, ratio_w, ratio_h)
        box_list,_ = box_list.convert()
        for i in range(len(box_list)):
            box_list[i] = [[box_list[i][0], box_list[i][4]],
                            [box_list[i][1], box_list[i][5]],
                            [box_list[i][2], box_list[i][6]],
                            [box_list[i][3], box_list[i][7]]]
        return np.array(box_list)

    def __call__(self, imgs):
        if isinstance(imgs, list):
            all_boxes = []
            for i, row in enumerate(imgs):
                for j, img in enumerate(row):
                    boxes = self.detect(img)
                    center = [(sum(box[:, :1]) / 4, sum(box[:, 1:2]) / 4) for box in boxes]
                    for ce, bo in zip(center, boxes):
                        correct = 1
                        buffx = self.bufferx / 2
                        buffy = self.buffery / 2
                        ebuffx = self.window_shape[0] - buffx
                        ebuffy = self.window_shape[1] - buffy
                        if ce[0] < buffx or ce[0] > ebuffx or ce[1] < buffy or ce[1] > ebuffy:
                            correct = 0

                        if i == 0 and ce[1] < buffy:
                            correct = 1
                        elif i == len(imgs) - 1 and ce[1] > ebuffy:
                            correct = 1
                        elif j == 0 and ce[0] < buffx:
                            correct = 1
                        elif j == len(row) and ce[0] > ebuffx:
                            correct = 1

                        if i == 0 and j == 0 and ce[0] < buffx and ce[1] < buffy:
                            correct = 1
                        elif i == 0 and j == len(row) and ce[0] > ebuffx and ce[1] < buffy:
                            correct = 1
                        elif i == len(imgs) and j == 0 and ce[0] < buffx and ce[1] > ebuffy:
                            correct = 1
                        elif i == len(imgs) and j == len(row) and ce[0] > ebuffx and ce[1] > ebuffy:
                            correct = 1
                        if correct == 1:
                            x_plus = (self.window_shape[0] - self.bufferx) * j
                            y_plus = (self.window_shape[1] - self.buffery) * i
                            bo = [[b[0] + x_plus, b[1] + y_plus] for b in bo]
                            all_boxes.append(np.array(bo))
            result = np.array(all_boxes)
        else:
            result = self.detect(imgs)
        return result

class Recoglayer():
    """
    A basic pipeline for performing recognition.

    Attributes
    ----------
    preprocess : callable
        the preprocess function whose input is a image
    recognizer : type of recognizer
        Recognizer subclass

    Methods
    -------
    __call__(image, boxes)
        image : crop image to recognize or big image with bounding boxes to crop and return list of recognized results
    """

    def __init__(self, model_path=None):
        super().__init__()
        parse = parse_base()
        args = parse.parse_args()
        self.cfg = setup("recog", args)
        if model_path is None:
            model_path = experiment_loader(type='recog')
        # model_path = "/home/bacnv6/projects/model_hub/akaocr/data/saved_models_recog/smz_recog/best_accuracy.pth"
        model = Atten(self.cfg)
        model = torch.nn.DataParallel(model).to(device)
        model.load_state_dict(torch.load(model_path, map_location=torch.device(device)), strict=False)
        model = model.to(device)

        self.recognizer = model
        self.converter = AttnLabelConverter(self.cfg["character"], device=device)
        self.max_h = self.cfg.MODEL.IMG_H
        self.max_w = self.cfg.MODEL.IMG_W
        self.pad = self.cfg.MODEL.PAD
        self.max_label_length = self.cfg.MODEL.MAX_LABEL_LENGTH
    
    def _remove_unknown(self, text):
        text = re.sub(f'[{self.cfg.SOLVER.UNKNOWN}]+', "", text)
        return text

    def recog(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if len(img.shape) == 2:  # (h, w)
            img = np.expand_dims(img, axis=-1)
        else:
            img = img
        if self.cfg.MODEL.INPUT_CHANNEL == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
        else:  # grayscale
            img = Image.fromarray(img[:, :, 0])
        align_collate = AlignCollate(img_h=self.max_h, img_w=self.max_w, keep_ratio_with_pad=self.pad)
        img_tensor = align_collate(img)
        with torch.no_grad():
            image = img_tensor.to(device)
            length_for_pred = torch.IntTensor([self.max_label_length]).to(device)
            text_for_pred = torch.LongTensor(1, self.max_label_length + 1).fill_(0).to(device)
            if 'Attn' in self.cfg.MODEL.PREDICTION:
                preds = self.recognizer(image, text_for_pred, is_train=False)
                # select max probability (greedy decoding) then decode index to character
                _, preds_index = preds.max(2)
                preds_str = self.converter.decode(preds_index, length_for_pred)

                preds_prob = F.softmax(preds, dim=2)
                preds_max_prob, _ = preds_prob.max(dim=2)

                pred_eos = preds_str[0].find('[s]')
                pred = preds_str[0][:pred_eos]  # prune after "end of sentence" token ([s])
                preds_max_prob = preds_max_prob[0][:pred_eos]
                # calculate confidence score (= multiply of pred_max_prob)
                try:
                    confidence_score = preds_max_prob.cumprod(dim=0)[-1]
                except:
                    confidence_score = 0
            else: 
                preds = self.recognizer(image, text_for_pred)

                # Select max probabilty (greedy decoding) then decode index to character
                preds_size = torch.IntTensor([preds.size(1)])
                _, preds_index = preds.max(2)
                preds_index = preds_index.view(-1)
                preds_str = self.converter.decode(preds_index.data, preds_size.data)

                pred = preds_str[0]
                preds_prob = F.softmax(preds, dim=2)
                preds_max_prob, _ = preds_prob.max(dim=2)

                # calculate confidence score (= multiply of pred_max_prob)
                try:
                    confidence_score = preds_max_prob.cumprod(dim=0)[-1]
                except:
                    confidence_score = 0  # for empty pred case, when prune after "end of sentence" token ([s])

            text = self._remove_unknown(pred)
            return text, confidence_score

    def __call__(self, img, boxes=None, output=None, seperator=None, fontpath=None):
        if output:
            if os.path.exists(output):
                shutil.rmtree(output)
            os.makedirs(output)
        if boxes is None:
            text, score = self.recog(img)
            if output:
                cv2.imwrite(os.path.join(output, text + '.jpg'), img)
            return text
        else:
            recog_result_list = list()
            confidence_score_list = list()
            for i, box in enumerate(boxes):
                poly = np.array(box).astype(np.int32)
                if len(poly.shape) == 1:
                    x0, y0 = poly[0], poly[1]
                    x1, y1 = poly[2], poly[3]
                else:
                    x0, y0 = np.min(poly, axis=0)
                    x1, y1 = np.max(poly, axis=0)
                roi = img[y0:y1, x0:x1]
                # print(roi)
                try:
                    if not seperator:
                        text, score = self.recog(roi)
                    elif seperator == 'logic':
                        _, l_roi = logic_seperator(roi)
                        text = ''
                        for ro in l_roi:
                            te, score = self.recog(ro)
                            text = text + te
                            # print(te)
                            # show_image(ro)
                except:
                    text = ''
                    print('cant recog box ', roi.shape)
                recog_result_list.append(text)
                confidence_score_list.append(score)
            if output and fontpath:
                vis = Visualizer(output_folder = output)
                img = vis.visualizer(image_ori=img, contours=boxes, font=fontpath, texts=recog_result_list)
                cv2.imwrite(os.path.join(output, "result.jpg"), img)
            return recog_result_list
