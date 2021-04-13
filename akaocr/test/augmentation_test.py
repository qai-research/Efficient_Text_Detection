import sys
sys.path.append("../")

from engine.config import setup, parse_base
from utils.file_utils import LmdbReader
from utils.augmentation import Augmentation
from utils.data import label_handler

def test_augmentation(args):
    cfg = setup("detec", args)
    option = {'shear':{'p':0.8, 'v':{'x':(-15,15), 'y':(-15,15)}},
            'scale':{'p':0.8, 'v':{"x": (0.8, 1.2), "y": (0.8, 1.2)}},
            'translate':{'p':0.8, 'v':{"x": (-0.2, 0.2), "y": (-0.2, 0.2)}},
            'rotate':{'p':0.8, 'v':(-45, 45)},
            'dropout':{'p':0.6,'v':(0.0, 0.5)},
            'blur'   :{'p':0.6,'v':(0.0, 4.0)},
            'elastic':{'p':0.85}}
    labelproc = label_handler.JsonLabelHandle()
    augmentation = Augmentation(cfg, option=option)
    lmdb = LmdbReader(args.detec_lmdb, cfg.MODEL.RGB)
    for i in range(1, args.num_samples):
        image, label = lmdb.get_item(i)
        label = labelproc(label)
        augmentation.augment([image], [label], imwrite=True, output_path = args.output_folder)

def main():
    parser = parse_base()
    parser.add_argument('--detec_lmdb', type=str, help='path to data')
    parser.add_argument('--num_samples', type=int, help='number of samples to test augment',default=2)
    parser.add_argument('--output_folder', type=str, help='path to output folder')
   
    args = parser.parse_args()
    test_augmentation(args)

if __name__ == '__main__':
    main()