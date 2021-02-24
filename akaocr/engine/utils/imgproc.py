import numpy as np
from skimage import io
import cv2


def load_image(img_file):
    img = io.imread(img_file)  # RGB order
    if img.shape[0] == 2:
        img = img[0]
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    img = np.array(img)

    return img


def normalize_mean_variance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    # should be RGB order
    img = in_img.copy().astype(np.float32)
    if len(img.shape) == 2:
        img -= np.array([mean[0] * 255.0], dtype=np.float32)
        img /= np.array([variance[0] * 255.0], dtype=np.float32)
    else:
        img -= np.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32)
        img /= np.array([variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0], dtype=np.float32)
    return img


def denormalize_mean_variance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    # should be RGB order
    img = in_img.copy()
    img *= variance
    img += mean
    img *= 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def resize_aspect_ratio(img, max_size, interpolation, mag_ratio=1):
    try:
        height, width, channel = img.shape
    except:
        height, width = img.shape
        channel = 1
# magnify image size
    target_size = mag_ratio * max(height, width)

    # set original image size
    if target_size > max_size:
        target_size = max_size

    ratio = target_size / max(height, width)

    target_h, target_w = int(height * ratio), int(width * ratio)
    proc = cv2.resize(img, (target_w, target_h), interpolation=interpolation)

    # make canvas and paste image
    target_h32, target_w32 = target_h, target_w
    if target_h % 32 != 0:
        target_h32 = target_h + (32 - target_h % 32)
    if target_w % 32 != 0:
        target_w32 = target_w + (32 - target_w % 32)
    resized = np.zeros((target_h32, target_w32, channel), dtype=np.float32)
    resized[0:target_h, 0:target_w, :] = proc
    target_h, target_w = target_h32, target_w32

    size_heatmap = (int(target_w / 2), int(target_h / 2))

    return resized, ratio, size_heatmap


def cvt2_heatmap_img(img):
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    return img


def main():
    img_path = '/home/aiteamdanang/aiteam/bacnv6/projects/akaocr/detec/craft/inputs/crop07_41_2.jpeg'
    img = load_image(img_path)
    print(img.shape)
    img, _, _ = resize_aspect_ratio(img, 2200, cv2.INTER_LINEAR)
    print(img.shape)


if __name__ == '__main__':
    main()
