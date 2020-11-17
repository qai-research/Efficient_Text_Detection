from math import exp
import numpy as np
import cv2
import os

# from . import imgproc


class GaussianTransformer(object):
    def __init__(self, img_size=512, region_threshold=0.4,
                 affinity_threshold=0.2):
        distance_ratio = 3.34
        scaled_gaussian = lambda x: exp(-(1 / 2) * (x ** 2))
        self.region_threshold = region_threshold
        self.imgSize = img_size
        self.standardGaussianHeat = self._gen_gaussian_heatmap(img_size, distance_ratio)

        _, binary = cv2.threshold(self.standardGaussianHeat, region_threshold * 255, 255, 0)
        np_contours = np.roll(np.array(np.where(binary != 0)), 1, axis=0).transpose().reshape(-1, 2)
        x, y, w, h = cv2.boundingRect(np_contours)
        self.regionbox = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.int32)
        # print("regionbox", self.regionbox)
        _, binary = cv2.threshold(self.standardGaussianHeat, affinity_threshold * 255, 255, 0)
        np_contours = np.roll(np.array(np.where(binary != 0)), 1, axis=0).transpose().reshape(-1, 2)
        x, y, w, h = cv2.boundingRect(np_contours)
        self.affinitybox = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.int32)
        # print("affinitybox", self.affinitybox)
        self.oribox = np.array([[0, 0, 1], [img_size - 1, 0, 1], [img_size - 1, img_size - 1, 1], [0, img_size - 1, 1]],
                               dtype=np.int32)

    @staticmethod
    def _gen_gaussian_heatmap(img_size, distance_ratio):
        scaled_gaussian = lambda x: exp(-(1 / 2) * (x ** 2))
        heat = np.zeros((img_size, img_size), np.uint8)
        for i in range(img_size):
            for j in range(img_size):
                distance_from_center = np.linalg.norm(np.array([i - img_size / 2, j - img_size / 2]))
                distance_from_center = distance_ratio * distance_from_center / (img_size / 2)
                scaled_gaussian_prob = scaled_gaussian(distance_from_center)
                heat[i, j] = np.clip(scaled_gaussian_prob * 255, 0, 255)
        return heat

    @staticmethod
    def _test():
        sigma = 10
        spread = 3
        extent = int(spread * sigma)
        center = spread * sigma / 2
        gaussian_heatmap = np.zeros([extent, extent], dtype=np.float32)

        for i_ in range(extent):
            for j_ in range(extent):
                gaussian_heatmap[i_, j_] = 1 / 2 / np.pi / (sigma ** 2) * np.exp(
                    -1 / 2 * ((i_ - center - 0.5) ** 2 + (j_ - center - 0.5) ** 2) / (sigma ** 2))

        gaussian_heatmap = (gaussian_heatmap / np.max(gaussian_heatmap) * 255).astype(np.uint8)
        images_folder = os.path.abspath(os.path.dirname(__file__)) + '/images'
        threshhold_guassian = cv2.applyColorMap(gaussian_heatmap, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(images_folder, 'test_guassian.jpg'), threshhold_guassian)

    def add_region_character(self, image, target_bbox, regionbox=None):
        if np.any(target_bbox < 0) or np.any(target_bbox[:, 0] > image.shape[1]) or np.any(
                target_bbox[:, 1] > image.shape[0]):
            return image
        affi = False
        if regionbox is None:
            regionbox = self.regionbox.copy()
        else:
            affi = True

        M = cv2.getPerspectiveTransform(np.float32(regionbox), np.float32(target_bbox))
        oribox = np.array(
            [[[0, 0], [self.imgSize - 1, 0], [self.imgSize - 1, self.imgSize - 1], [0, self.imgSize - 1]]],
            dtype=np.float32)
        test1 = cv2.perspectiveTransform(np.array([regionbox], np.float32), M)[0]
        real_target_box = cv2.perspectiveTransform(oribox, M)[0]
        # print("test\ntarget_bbox", target_bbox, "\ntest1", test1, "\nreal_target_box", real_target_box)
        real_target_box = np.int32(real_target_box)
        real_target_box[:, 0] = np.clip(real_target_box[:, 0], 0, image.shape[1])
        real_target_box[:, 1] = np.clip(real_target_box[:, 1], 0, image.shape[0])

        # warped = cv2.warpPerspective(self.standardGaussianHeat.copy(), M, (image.shape[1], image.shape[0]))
        # warped = np.array(warped, np.uint8)
        # image = np.where(warped > image, warped, image)
        if np.any(target_bbox[0] < real_target_box[0]) or (
                target_bbox[3, 0] < real_target_box[3, 0] or target_bbox[3, 1] > real_target_box[3, 1]) or (
                target_bbox[1, 0] > real_target_box[1, 0] or target_bbox[1, 1] < real_target_box[1, 1]) or (
                target_bbox[2, 0] > real_target_box[2, 0] or target_bbox[2, 1] > real_target_box[2, 1]):
            # if False:
            warped = cv2.warpPerspective(self.standardGaussianHeat.copy(), M, (image.shape[1], image.shape[0]))
            warped = np.array(warped, np.uint8)
            image = np.where(warped > image, warped, image)
            # _M = cv2.getPerspectiveTransform(np.float32(regionbox), np.float32(_target_box))
            # warped = cv2.warpPerspective(self.standardGaussianHeat.copy(), _M, (width, height))
            # warped = np.array(warped, np.uint8)
            #
            # # if affi:
            # # print("warped", warped.shape, real_target_box, target_bbox, _target_box)
            # # cv2.imshow("1123", warped)
            # # cv2.waitKey()
            # image[ymin:ymax, xmin:xmax] = np.where(warped > image[ymin:ymax, xmin:xmax], warped,
            #                                        image[ymin:ymax, xmin:xmax])
        else:
            xmin = real_target_box[:, 0].min()
            xmax = real_target_box[:, 0].max()
            ymin = real_target_box[:, 1].min()
            ymax = real_target_box[:, 1].max()

            width = xmax - xmin
            height = ymax - ymin
            _target_box = target_bbox.copy()
            _target_box[:, 0] -= xmin
            _target_box[:, 1] -= ymin
            _M = cv2.getPerspectiveTransform(np.float32(regionbox), np.float32(_target_box))
            warped = cv2.warpPerspective(self.standardGaussianHeat.copy(), _M, (width, height))
            warped = np.array(warped, np.uint8)
            if warped.shape[0] != (ymax - ymin) or warped.shape[1] != (xmax - xmin):
                print("region (%d:%d,%d:%d) warped shape (%d,%d)" % (
                    ymin, ymax, xmin, xmax, warped.shape[1], warped.shape[0]))
                return image
            # if affi:
            # print("warped", warped.shape, real_target_box, target_bbox, _target_box)
            # cv2.imshow("1123", warped)
            # cv2.waitKey()
            image[ymin:ymax, xmin:xmax] = np.where(warped > image[ymin:ymax, xmin:xmax], warped,
                                                   image[ymin:ymax, xmin:xmax])
        return image

    def add_affinity_character(self, image, target_bbox):
        return self.add_region_character(image, target_bbox, self.affinitybox)

    def add_affinity(self, image, bbox_1, bbox_2):
        center_1, center_2 = np.mean(bbox_1, axis=0), np.mean(bbox_2, axis=0)
        tl = np.mean([bbox_1[0], bbox_1[1], center_1], axis=0)
        bl = np.mean([bbox_1[2], bbox_1[3], center_1], axis=0)
        tr = np.mean([bbox_2[0], bbox_2[1], center_2], axis=0)
        br = np.mean([bbox_2[2], bbox_2[3], center_2], axis=0)

        affinity = np.array([tl, tr, br, bl])

        return self.add_affinity_character(image, affinity.copy()), np.expand_dims(affinity, axis=0)

    def generate_region(self, image_size, bboxes):
        height, width = image_size[0], image_size[1]
        target = np.zeros([height, width], dtype=np.uint8)
        for i in range(len(bboxes)):
            character_bbox = np.array(bboxes[i].copy())
            for j in range(bboxes[i].shape[0]):
                target = self.add_region_character(target, character_bbox[j])

        return target

    def generate_affinity(self, image_size, bboxes, words):
        height, width = image_size[0], image_size[1]
        target = np.zeros([height, width], dtype=np.uint8)
        affinities = []
        for i in range(len(words)):
            character_bbox = np.array(bboxes[i])
            total_letters = 0
            for char_num in range(character_bbox.shape[0] - 1):
                target, affinity = self.add_affinity(target, character_bbox[total_letters],
                                                     character_bbox[total_letters + 1])
                affinities.append(affinity)
                total_letters += 1
        if len(affinities) > 0:
            affinities = np.concatenate(affinities, axis=0)
        return target, affinities

    def save_gaussian_heat(self):
        images_folder = os.path.abspath(os.path.dirname(__file__)) + '/images'
        cv2.imwrite(os.path.join(images_folder, 'standard.jpg'), self.standardGaussianHeat)
        warped_color = cv2.applyColorMap(self.standardGaussianHeat, cv2.COLORMAP_JET)
        cv2.polylines(warped_color, [np.reshape(self.regionbox, (-1, 1, 2))], True, (255, 255, 255), thickness=1)
        cv2.imwrite(os.path.join(images_folder, 'standard_color.jpg'), warped_color)
        standard_gaussian_heat1 = self.standardGaussianHeat.copy()
        threshhold = self.region_threshold * 255
        standard_gaussian_heat1[standard_gaussian_heat1 > 0] = 255
        threshhold_guassian = cv2.applyColorMap(standard_gaussian_heat1, cv2.COLORMAP_JET)
        cv2.polylines(threshhold_guassian, [np.reshape(self.regionbox, (-1, 1, 2))], True, (255, 255, 255), thickness=1)
        cv2.imwrite(os.path.join(images_folder, 'threshhold_guassian.jpg'), threshhold_guassian)


def main():
    gaussian = GaussianTransformer(512, 0.4, 0.2)
    gaussian.save_gaussian_heat()
    gaussian._test()
    bbox0 = np.array([[[0, 0], [100, 0], [100, 100], [0, 100]]])
    image = np.zeros((500, 500), np.uint8)
    # image = gaussian.add_region_character(image, bbox)
    bbox1 = np.array([[[100, 0], [200, 0], [200, 100], [100, 100]]])
    bbox2 = np.array([[[100, 100], [200, 100], [200, 200], [100, 200]]])
    bbox3 = np.array([[[0, 100], [100, 100], [100, 200], [0, 200]]])

    bbox4 = np.array([[[96, 0], [151, 9], [139, 64], [83, 58]]])
    # image = gaussian.add_region_character(image, bbox)
    # print(image.max())
    image = gaussian.generate_region((500, 500, 1), [bbox4])
    target_gaussian_heatmap_color = imgproc.cvt2_heatmap_img(image.copy() / 255)
    cv2.imshow("test", target_gaussian_heatmap_color)
    cv2.imwrite("test.jpg", target_gaussian_heatmap_color)
    cv2.waitKey()
    # weight, target = gaussian.generate_target((1024, 1024, 3), bbox.copy())
    # target_gaussian_heatmap_color = imgproc.cvt2HeatmapImg(weight.copy() / 255)
    # cv2.imshow('test', target_gaussian_heatmap_color)
    # cv2.waitKey()
    # cv2.imwrite("test.jpg", target_gaussian_heatmap_color)


if __name__ == '__main__':
    main()
