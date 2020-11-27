import sys

sys.path.append("../")
import os
import torch
import numpy as np
import cv2

from models.detec.heatmap import HEAT
from models.recog.atten import Atten
from models.modules.converters import AttnLabelConverter
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test_model_detec():
    model = HEAT()

    model = model.to(device)
    x = torch.randn(1, 3, 768, 768).to(device)
    print(x.shape)
    y = model(x)
    print(y[0].shape)
    print(y[1].shape)


def test_model_recog():
    config = dict()
    config["transformation"] = "TPS"
    config["feature_extraction"] = "ResNet"
    config["sequence_modeling"] = "BiLSTM"
    config["prediction"] = "Attn"
    config["beam_size"] = 1
    config["num_fiducial"] = 20
    config["input_channel"] = 1
    config["output_channel"] = 512
    config["hidden_size"] = 128
    config["img_h"] = 32
    config["img_w"] = 128
    config["device"] = None
    config["num_class"] = 3120
    config["max_label_length"] = 15
    model = Atten(config)

    x = torch.randn(1, 1, 32, 128)
    # print(x.shape)
    text = ["xxx"]
    converter = AttnLabelConverter(["x", "X", "o"], device=config["device"])
    text, length = converter.encode(text, max_label_length=config["max_label_length"])
    y = model(x, text)
    print(y.shape)
    # print(y)


if __name__ == '__main__':
    test_model_detec()
    test_model_recog()