# import sys
# sys.path.append("../")
# from engine.config import parse_base
# from pipeline.layers import SlideWindow, Detectlayer, Recoglayer
# import cv2

# def test_pipeline(args):
#     img = cv2.imread(args.img_path)
#     slidewindow = SlideWindow(window=(1280, 800))
#     deteclayer = Detectlayer(args=args, model_path = args.w_detec)
#     recoglayer = Recoglayer(args=args, model_path = args.w_recog)
#     out = slidewindow(img)
#     out = deteclayer(img)
#     out = recoglayer(img, boxes=out, output=args.output_path, fontpath=args.font_path)

# def main():
#     parser = parse_base()
#     parser.add_argument('--w_detec', type=str, help='path to detec model .pth')
#     parser.add_argument('--w_recog', type=str, help='path to recog model .pth')
#     parser.add_argument('--font_path', type=str, help='path to font .ttf')
#     parser.add_argument('--img_path', type=str, help='path to image .jpg')
#     parser.add_argument('--output_path', type=str, help='path to output folder')
#     args = parser.parse_args()

#     test_pipeline(args)

# if __name__ == '__main__':
#     main()

import sys
sys.path.append("../")
from pipeline.layers import SlideWindow, Detectlayer, Recoglayer
import cv2

img_path = "input/image_1.jpg"
img = cv2.imread(img_path)
output_path = "/home/bacnv6/kimnh3/tmp/"
fontpath = "input/default_vis_font.ttf" 

model_detec_path = "/home/bacnv6/kimnh3/data_test_ci/saved_models_detec/best_accuracy.pth"
model_recog_path = "/home/bacnv6/kimnh3/data_test_ci/saved_models_recog/best_accuracy.pth"

slidewindow = SlideWindow(window=(1280, 800))
deteclayer = Detectlayer(model_path = model_detec_path)
recoglayer = Recoglayer(model_path = model_recog_path)
out = slidewindow(img)
out = deteclayer(img)
out = recoglayer(img, boxes=out, output=output_path, fontpath=fontpath)

#python pipeline_test.py --w_detec=/home/bacnv6/kimnh3/data_test_ci/saved_models_detec/best_accuracy.pth 
# --w_recog=/home/bacnv6/kimnh3/data_test_ci/saved_models_recog/best_accuracy.pth 
# --font_path=input/default_vis_font.ttf --img_path=input/image_1.jpg --output_path=/home/bacnv6/kimnh3/tmp/