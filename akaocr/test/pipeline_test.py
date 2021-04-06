import sys
sys.path.append("../")
from pipeline.layers import SlideWindow, Detectlayer, Recoglayer
import cv2

img_path = "/home/aic/nghiann3/image_1.jpg"
img = cv2.imread(img_path)
output_path = "/home/aic/nghiann3/output/"
fontpath = "/home/bacnv6/projects/model_hub/akaocr/data/default_vis_font.ttf" 

model_detec_path = "/home/bacnv6/projects/model_hub/akaocr/data/saved_models_detec/smz_detec/best_accuracy.pth"
model_recog_path = "/home/bacnv6/projects/model_hub/akaocr/data/saved_models_recog/smz_recog/best_accuracy.pth"

slidewindow = SlideWindow(window=(1280, 800))
deteclayer = Detectlayer(model_path = model_detec_path)
recoglayer = Recoglayer(model_path = model_recog_path)
out = slidewindow(img)
out = deteclayer(img)
out = recoglayer(img, boxes=out, output=output_path, fontpath=fontpath)
