import sys
sys.path.append("../")
from engine.config import parse_base
from pipeline.layers import SlideWindow, Detectlayer, Recoglayer
import cv2

def test_pipeline(args):
    img = cv2.imread(args.img_path)
    slidewindow = SlideWindow(window=(1280, 800))
    deteclayer = Detectlayer(args=args, model_path = args.w_detec)
    recoglayer = Recoglayer(args=args, model_path = args.w_recog)
    out = slidewindow(img)
    out = deteclayer(img)
    out = recoglayer(img, boxes=out, output=args.output_path, fontpath=args.font_path)

def main():
    parser = parse_base()
    parser.add_argument('--w_detec', type=str, help='path to detec model .pth')
    parser.add_argument('--w_recog', type=str, help='path to recog model .pth')
    parser.add_argument('--font_path', type=str, help='path to font .ttf')
    parser.add_argument('--img_path', type=str, help='path to image .jpg')
    parser.add_argument('--output_path', type=str, help='path to output folder')
    args = parser.parse_args()

    test_pipeline(args)

if __name__ == '__main__':
    main()
