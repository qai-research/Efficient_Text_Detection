import sys
sys.path.append("../")
from models.detec.heatmap import HEAT
from models.recog.atten import Atten
import torch
import cv2
from utils.utility import initial_logger
logger = initial_logger()

from engine.metric.evaluation import Evaluation
from engine.config import setup
from utils.data.dataloader import load_test_dataset_detec

def detec_test_evaluation(model_path, data_path):
    cfg = setup("detec")
    model = HEAT()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    test_loader = load_test_dataset_detec(data_path)
    evaluate = Evaluation(cfg, model, test_loader, num_samples=4)
    evaluate.detec_evaluation()
    
def recog_test_evaluation(model_path, data_path):
    pass

if __name__=='__main__':
    model_path = '/home/nghianguyen/smz_detec/best_accuracy.pth'
    data_path = '/home/nghianguyen/train_data/lake_detec/ST_Demo_1'
    detec_test_evaluation(model_path, data_path)