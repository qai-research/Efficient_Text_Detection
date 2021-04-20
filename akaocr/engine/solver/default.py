from yacs.config import CfgNode as CN
"""
Borrow detectron2 config method for optimizers and learning rate schedulers
"""

_C = CN()

##################
#################
_C._BASE_ = CN()
_C._BASE_.MANUAL_SEED = 1111
_C._BASE_.MODEL_TYPE = None
####################
# ---------------------------------------------------------------------------- #
# MODEL
# ---------------------------------------------------------------------------- #
_C.MODEL = CN()
###### config model for recog
# whether to do fine-tuning only the last layer or not, bool
_C.MODEL.FT = False
# the height of the input image to rectifier', default=64
_C.MODEL.IMG_H = 32
# the width of the input image to rectifier', default=200
_C.MODEL.IMG_W = 128
# for sensitive character mode
_C.MODEL.SENSITIVE = True
# whether to keep ratio then pad for image resize
_C.MODEL.PAD = True
# type of image input rgb or gray
_C.MODEL.RGB = True
# Maximum length for words
_C.MODEL.MAX_LABEL_LENGTH = 15
# Transformation stage. None|TPS
_C.MODEL.TRANSFORMATION = None
# Feature Extraction stage. ResNet
_C.MODEL.FEATURE_EXTRACTION = None
# SequenceModeling stage. None|BiLSTM
_C.MODEL.SEQUENCE_MODELING = None
# Prediction stage. CTC|Attn
_C.MODEL.PREDICTION = None
# number of fiducial points of TPS-STN
_C.MODEL.NUM_FIDUCIAL = 20
# the number of input channel of Feature extractor
_C.MODEL.INPUT_CHANNEL = 1
# the number of output channel of Feature extractor
_C.MODEL.OUTPUT_CHANNEL = 512
# the size of the LSTM hidden state
_C.MODEL.HIDDEN_SIZE = 128
# name of the vocab file
_C.MODEL.VOCAB = None
#########config model for detec
# name of model detec (CRAFT, RESNET, EFFICIENT)
_C.MODEL.NAME = "CRAFT"
# the minimum size of the image's longer dimension, int
_C.MODEL.MIN_SIZE = 700
# the maximum size of the image's longer dimension, int
_C.MODEL.MAX_SIZE = 1200
# type of image input rgb or gray
_C.MODEL.RGB = True
###################
###################
# ---------------------------------------------------------------------------- #
# INFERENCE
# ---------------------------------------------------------------------------- #
_C.INFERENCE = CN()
# threshold for texts heat map, 0 < text_threshold < 1, smaller means fewer text filtered, float
_C.INFERENCE.TEXT_THRESHOLD = 0.7

# threshold for affinities heat map between characters, 0 < link_threshold < 1, bigger means fewer link filtered, float
_C.INFERENCE.LINK_THRESHOLD = 0.4

# threshold for accept a word to be detected, 0 < text_threshold < 1, smaller means fewer text filtered, float
_C.INFERENCE.LOW_TEXT_SCORE = 0.4

####################
####################
# ---------------------------------------------------------------------------- #
# VERSION
# ---------------------------------------------------------------------------- #
_C.VERSION = None

####################
####################
# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
####################
# beam size for ocr
_C.SOLVER.BEAM_SIZE = 1
# for data_filtering mode
_C.SOLVER.DATA_FILTERING = True
# replace all OOV characters with this character
_C.SOLVER.UNKNOWN = ''

####################
# number of data loading workers default=0, int
_C.SOLVER.WORKERS = 4
# maximum number of iteration
_C.SOLVER.MAX_ITER = 300000
# interval to do validation
_C.SOLVER.EVAL_PERIOD = 5000
# IDs of devices that host the data, model, etc.
_C.SOLVER.DEVICE_IDS = None
# batch size of training default=1, int
_C.SOLVER.BATCH_SIZE = 1
# initial learning rate, float
_C.SOLVER.LR = 3.2768e-3

############################
# See detectron2/solver/build.py for LR scheduler options
# List of LR scheduler options from detectron2 [WarmupMultiStepLR, WarmupCosineLR]
# List of LR scheduler options from pytorch [CosineAnnealingLR, ReduceLROnPlateau, CosineAnnealingWarmRestarts]
# List of LR scheduler options from akaOCR [WarmupDecayCosineLR]
_C.SOLVER.LR_SCHEDULER_NAME = "WarmupDecayCosineLR"

_C.SOLVER.MAX_ITER = 40000

_C.SOLVER.BASE_LR = 0.001

_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.NESTEROV = False

_C.SOLVER.WEIGHT_DECAY = 0.0001
# The weight decay that's applied to parameters of normalization layers
# (typically the affine transformation)
_C.SOLVER.WEIGHT_DECAY_NORM = 0.0

_C.SOLVER.GAMMA = 0.1
# The iteration number to decrease learning rate by GAMMA.
_C.SOLVER.STEPS = (30000, 60000, 90000, 120000, 150000)

_C.SOLVER.WARMUP_FACTOR = 1.0 / 1000
_C.SOLVER.WARMUP_ITERS = 1000
_C.SOLVER.WARMUP_METHOD = "linear"

# Save a checkpoint after every this number of iterations
_C.SOLVER.CHECKPOINT_PERIOD = 5000
# _C.SOLVER.CHECKPOINT_PERIOD = 500
# Number of images per batch across all machines.
# If we have 16 GPUs and IMS_PER_BATCH = 32,
# each GPU will see 2 images per batch.
# May be adjusted automatically if REFERENCE_WORLD_SIZE is set.
# _C.SOLVER.IMS_PER_BATCH = 16

# The reference number of workers (GPUs) this config is meant to train with.
# It takes no effect when set to 0.
# With a non-zero value, it will be used by DefaultTrainer to compute a desired
# per-worker batch size, and then scale the other related configs (total batch size,
# learning rate, etc) to match the per-worker batch size.
# See documentation of `DefaultTrainer.auto_scale_workers` for details:
_C.SOLVER.REFERENCE_WORLD_SIZE = 0

# Detectron v1 (and previous detection code) used a 2x higher LR and 0 WD for
# biases. This is not useful (at least for recent models). You should avoid
# changing these and they exist only to reproduce Detectron v1 training if
# desired.
_C.SOLVER.BIAS_LR_FACTOR = 1.0
_C.SOLVER.WEIGHT_DECAY_BIAS = _C.SOLVER.WEIGHT_DECAY

# Gradient clipping
# _C.SOLVER.CLIP_GRADIENTS = CN()
_C.SOLVER.CLIP_GRADIENTS = CN({"ENABLED": False})

# Type of gradient clipping, currently 2 values are supported:
# - "value": the absolute values of elements of each gradients are clipped
# - "norm": the norm of the gradient for each parameter is clipped thus
#   affecting all elements in the parameter
_C.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "value"
# Maximum absolute value used for clipping gradients
_C.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0
# Floating point number p for L-p norm to be used with the "norm"
# gradient clipping type; for L-inf, please specify .inf
_C.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0

# Enable automatic mixed precision for training
# Note that this does not change model's inference behavior.
# To use AMP in inference, run inference under autocast()
# _C.SOLVER.AMP = False

_C.SOLVER.NUM_SAMPLES = 50
# _C.SOLVER.NUM_SAMPLES = 200
# _C.SOLVER.NUM_SAMPLES = 1000
_C.SOLVER.EARLY_STOP_AFTER = 5