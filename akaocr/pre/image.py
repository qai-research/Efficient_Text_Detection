import cv2
import numpy as np
from skimage import io
from pathlib import Path


class ImageProc:
    """
    Contain image pre process functions
    """
    @staticmethod
    def binarize(image, maxval=255):
        """
        Binarization by ostu threshold
        Parameters
        ----------
        image : numpy array
            the image to be processed
        maxval : int, optional, default: 255
            the maximum value of an image pixel

        Returns
        ------
        numpy array, shape (height, width)
            the processed image
        """
        img = cv2.threshold(image, 0, maxval, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        return img

    @staticmethod
    def morph(image, size=(3, 3), mode='open', iteration=1):
        """
        Morphology transform
        Parameters
        ----------
        image : 2D numpy array
            the image to be processed
        size : tuple, optional, default: (3, 3)
            the size of the filter kernel
        mode : "open" or "close", optional, default: "open"
            whether to perform "open" morphological transformation or "close"
        iteration : int, optional, default: 1
            how many times to run the transform

        Returns
        ------
        2D numpy array, shape (height, width)
            the processed image
        """
        kernel = np.ones(size, np.uint8)
        if mode == "open":  # noise inside
            img = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=iteration)
        elif mode == "close":  # noise outside
            img = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iteration)
        else:
            img = image
        return img

    @staticmethod
    def noise_reduce(image, maxval=255, thres_area=0):
        """
        Clean small blobs noise in image
        Parameters
        ----------
        image : numpy array
            the image to be processed
        maxval : int, optional, default: 255
            the maximum value of an image pixel
        thres_area : float, optional, default: 0
            the minimum area of a blob to be considered as not noise

        Returns
        ------
        numpy array
            the processed image
        """
        # invert image
        img = maxval - image
        _, contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.ones(img.shape, dtype=img.dtype) * maxval  # all white

        for c in contours:
            area = cv2.contourArea(c)
            if area < thres_area:
                cv2.drawContours(mask, [c], -1, (0, 0, 0), -1)  # 0 out meaning remove

        mask = mask / maxval
        img = img * mask
        img = maxval - img
        return img

    @staticmethod
    def clean_space(image_ori, empty_thres=200, text_color='black', clean_thres=20, noise_thres=7):
        """
        Clean empty space in croped text before recognition
        Parameters
        ----------
        image_ori : numpy array
            the image to be processed
        empty_thres : int, optional, default: 200
            maximum value of image density in x direction
        text_color : str, optional, default: 'black'
            the the color of text on image
        clean_thres : int, optional, default: 20
            the maximum allow separation between text
        noise_thres : int, optional, default: 7
            the threshold for a slide be consider as text

        Returns
        ------
        numpy array
            the processed image
        """

        image = image_ori.copy()
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if text_color == 'black':
            image = cv2.bitwise_not(image)
        hist = np.sum(image, 0)
        empty_space = np.zeros(hist.shape)
        list_empty_lot = []
        empty_lot = []
        for i, h in enumerate(hist):
            if h < empty_thres:
                empty_space[i] = 1
                empty_lot.append(i)
            else:
                empty_space[i] = 0
                if np.sum(hist[i:i + 20]) > noise_thres:
                    list_empty_lot.append(empty_lot)
                    empty_lot = []
                else:
                    empty_lot.append(i)
        list_empty_lot = [li for li in list_empty_lot if len(li) > 0]
        list_empty_lot.reverse()
        for i, lot in enumerate(list_empty_lot):
            llot = len(lot)
            if llot > clean_thres:
                num_clean_pixels = llot - clean_thres
                start_clean_from = int(clean_thres / 2)
                stop_clean_in = start_clean_from + num_clean_pixels
                list_clean = lot[start_clean_from:stop_clean_in]
                image_ori = np.delete(image_ori, list_clean, 1)
        return image_ori

    @staticmethod
    def resize_aspect_ratio(image, max_size, interpolation=cv2.INTER_LINEAR):
        """
        Resize image bigger edge to max_size
        :param image: numpy array
        :param max_size: new max side
        :param interpolation: type of interpolation
        :return:
        """
        height, width = image.shape[:2]
        ratio = max_size / max(height, width)

        target_h, target_w = int(height * ratio), int(width * ratio)
        image = cv2.resize(image, (target_w, target_h), interpolation=interpolation)
        return image, ratio

    # @staticmethod
    def random_scale(self, image, min_size=768, max_size=1280):
        """
        Random resize image to the range of min_size to max_side
        :param image: numpy array
        :param min_size: min to resize
        :param max_size: max to resize
        :return: resized image
        """
        size = random.randint(min_size, max_size)
        image, ratio = self.resize_aspect_ratio(image, size)
        return image, ratio

    @staticmethod
    def load_image(img_file):
        """
        read image from path
        :param img_file: path to image
        :return: image: numpy array
        """
        img = io.imread(img_file)  # RGB order
        if img.shape[0] == 2:
            img = img[0]
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if img.shape[2] == 4:
            img = img[:, :, :3]
        img = np.array(img)
        return img

    @staticmethod
    def normalize_mean_variance(image, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
        """
        Normalize image to fix mean and variance
        :param image: numpy array
        :param mean: mean value for channels
        :param variance: variance value for channels
        :return: normalized image
        """
        # should be RGB order
        img = image.copy().astype(np.float32)
        if len(img.shape) == 2:
            img -= np.array([mean[0] * 255.0], dtype=np.float32)
            img /= np.array([variance[0] * 255.0], dtype=np.float32)
        else:
            img -= np.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32)
            img /= np.array([variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0], dtype=np.float32)
        return img

    @staticmethod
    def denormalize_mean_variance(image, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
        """
        Denormalize image from previous transformed mean and variance
        :param image: numpy array
        :param mean:
        :param variance:
        :return: original image
        """
        # should be RGB order
        img = image.copy()
        if len(img.shape) == 2:
            img *= variance[0]
            img += mean[0]
            img *= 255.0
            img = np.clip(img, 0, 255).astype(np.uint8)
        else:
            img *= variance
            img += mean
            img *= 255.0
            img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    @staticmethod
    def cvt2_heatmap_img(heatmap):
        """
        Generate visualize for heat map
        :param heatmap: heatmap matrix with value between 0->1
        :return: visualize heatmap image
        """
        heatmap = (np.clip(heatmap, 0, 1) * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        return heatmap

    def save_heatmap(self, image_name, image, bboxes, affinity_bboxes, region_scores,
                     affinity_scores, confidence_mask, output_path="output"):
        output_image = np.uint8(image.copy())
        output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
        if len(bboxes) > 0:
            affinity_bboxes = np.int32(affinity_bboxes)
            for i in range(affinity_bboxes.shape[0]):
                cv2.polylines(output_image, [np.reshape(affinity_bboxes[i], (-1, 1, 2))], True, (255, 0, 0))
            for i in range(len(bboxes)):
                _bboxes = np.int32(bboxes[i])
                for j in range(_bboxes.shape[0]):
                    cv2.polylines(output_image, [np.reshape(_bboxes[j], (-1, 1, 2))], True, (0, 0, 255))

        target_gaussian_heatmap_color = self.cvt2_heatmap_img(region_scores / 255)
        target_gaussian_affinity_heatmap_color = self.cvt2_heatmap_img(affinity_scores / 255)
        heat_map = np.concatenate([target_gaussian_heatmap_color, target_gaussian_affinity_heatmap_color], axis=1)
        confidence_mask_gray = self.cvt2_heatmap_img(confidence_mask)
        output = np.concatenate([output_image, heat_map, confidence_mask_gray], axis=1)
        out_path = Path(output_path).joinpath("%s_input.jpg" % Path(image_name).stem)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), output)

    @staticmethod
    def sub_image_random_crop(imgs, img_size, character_bboxes):

        """
        Random crop a small fix size image from the original image
        :param imgs: list of input image to crop
        :param img_size: design output size
        :param character_bboxes: list of character boxes in image
        :return:
        """
        h, w = imgs[0].shape[0:2]
        th, tw = img_size
        if w == tw and h == th:
            return imgs

        word_bboxes = []
        if len(character_bboxes) > 0:
            for bboxes in character_bboxes:
                word_bboxes.append(
                    [[bboxes[:, :, 0].min(), bboxes[:, :, 1].min()], [bboxes[:, :, 0].max(), bboxes[:, :, 1].max()]])
        word_bboxes = np.array(word_bboxes, np.int32)

        # IC15 for 0.6, MLT for 0.35
        if random.random() > 0.6 and len(word_bboxes) > 0:
            sample_bboxes = word_bboxes[random.randint(0, len(word_bboxes) - 1)]
            left = max(sample_bboxes[1, 0] - img_size[0], 0)
            top = max(sample_bboxes[1, 1] - img_size[0], 0)

            if min(sample_bboxes[0, 1], h - th) < top or min(sample_bboxes[0, 0], w - tw) < left:
                i = random.randint(0, h - th)
                j = random.randint(0, w - tw)
            else:
                i = random.randint(top, min(sample_bboxes[0, 1], h - th))
                j = random.randint(left, min(sample_bboxes[0, 0], w - tw))

            crop_h = sample_bboxes[1, 1] if th < sample_bboxes[1, 1] - i else th
            crop_w = sample_bboxes[1, 0] if tw < sample_bboxes[1, 0] - j else tw
        else:
            # train for MLT dataset
            i, j = 0, 0
            crop_h, crop_w = h + 1, w + 1  # make the crop_h, crop_w > tw, th

        for idx in range(len(imgs)):
            if len(imgs[idx].shape) == 3:
                imgs[idx] = imgs[idx][i:i + crop_h, j:j + crop_w, :]
            else:
                imgs[idx] = imgs[idx][i:i + crop_h, j:j + crop_w]

            if crop_w > tw or crop_h > th:
                imgs[idx] = padding_image(imgs[idx], tw)
        return imgs

    @staticmethod
    def resize_gt(gt_mask, size):
        """
        Resize the matrix to square of (side*side)
        :param gt_mask: original matrix
        :param size: square side
        :return: resized image
        """
        return cv2.resize(gt_mask, (size, size))
