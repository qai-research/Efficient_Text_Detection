import sys
sys.path.append("../")

from engine.metric import tedeval
from engine.infer.heat2boxes import Heat2boxes
from pre.image import ImageProc
import torch
import cv2

class Evaluation:
    """This module contains evaluation methods for detec and recog models"""

    def __init__(self, cfg, model, test_loader, num_samples = None):
        """
        Args:
            model: model for evaluation
            test_loader: data for evaluation
            num_samples: number of sample will be evaluated
        """

        self.cfg = cfg
        self.max_size = self.cfg.MODEL.MAX_SIZE
        self.model = model
        self.test_loader = test_loader
        if num_samples is None:
            self.num_samples = self.test_loader.get_length()
        else:
            self.num_samples = num_samples

    """Evaluate detec model"""
    def detec_evaluation(self):
        recall = 0
        precision = 0
        hmean = 0
        AP = 0
        for i in range(1, self.num_samples+1):
            pre_box_list = list()       #list of predicted boxes from model
            gt_box_list = list()        #list of ground truth boxes in labels
            word_list = list()          #list of words in labels
            img, label = self.test_loader.get_item(i)
            gt_box = list()
            words = list()
            for j in range(len(label['words'])):
                words.append(label['words'][j]['text'])
                x1 = label['words'][j]['x1']
                x2 = label['words'][j]['x2']
                x3 = label['words'][j]['x3']
                x4 = label['words'][j]['x4']
                y1 = label['words'][j]['y1']
                y2 = label['words'][j]['y2']
                y3 = label['words'][j]['y3']
                y4 = label['words'][j]['y4']
                box = [x1, y1, x2, y2, x3, y3, x4, y4]
                gt_box.append(box)
    
            img, target_ratio = ImageProc.resize_aspect_ratio(
                img, self.max_size, interpolation=cv2.INTER_LINEAR
            )
            ratio_h = ratio_w = 1 / target_ratio
            img = ImageProc.normalize_mean_variance(img)
            img = torch.from_numpy(img).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
            img = (img.unsqueeze(0))  # [c, h, w] to [b, c, h, w]
            y,_ = self.model(img)
            del img     # delete variable img to save memory
            box_list = Heat2boxes(self.cfg, y, ratio_w, ratio_h)
            del y       # delete variable y to save memory
            box_list,_ = box_list.convert(evaluation=True)

            pre_box_list.append(box_list)
            gt_box_list.append(gt_box)
            word_list.append(words)
            
            confidence_point_list = list()            
            for k in range(len(pre_box_list)):
                confidence_point = list()
                for j in range(len(pre_box_list[k])):
                    confidence_point.append(0.0)
                confidence_point_list.append(confidence_point)
            detec_eval = tedeval.Evaluate(pre_box_list, gt_box_list, word_list, confidence_point_list)
            resdict = detec_eval.do_eval()
            recall += resdict['method']['recall']
            precision += resdict['method']['precision']
            hmean += resdict['method']['hmean']
            AP += resdict['method']['AP']
        print('recall:', recall/self.num_samples)
        print('precision:', precision/self.num_samples)
        print('hmean:', hmean/self.num_samples)
        print('AP:', AP/self.num_samples)
    """Evaluate recog model"""
    def recog_evaluation(self):
        pass
