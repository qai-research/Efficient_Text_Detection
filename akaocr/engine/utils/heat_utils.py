import numpy as np
import cv2
import math


# unwarp coordinates
def warp_coord(min_v, pt):
    out = np.matmul(min_v, (pt[0], pt[1], 1))
    return np.array([out[0] / out[2], out[1] / out[2]])


def show_image(img, boxes=None, lines=None, contours=None, name='demo', windows=(1000, 1000), show=1):
    if boxes is not None:
        for b in boxes:
            # print(b)
            img = cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (255, 0, 0), 1)
    if lines is not None:
        for li in lines:
            li = li[0]
            img = cv2.line(img, (li[0], li[1]), (li[2], li[3]), (255, 0, 0), 2, cv2.LINE_AA)

    if contours is not None:
        # img = cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
        img = cv2.drawContours(img, contours.astype(int), -1, (0, 255, 0), 2)

    if show == 1:
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, windows[0], windows[1])
        cv2.imshow(name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return img


def height_spliter(element):
    scale = 1.5
    ignore_threshold = 15

    list_element = list()
    temp_element = list()
    current_height = -1
    for i, el in enumerate(element):
        # print(el, len(temp_element))
        if len(temp_element) == 0 and i == len(element)-1:
            # print("start-end", current_height)
            temp_element.append(el)
            list_element.append(temp_element)
        elif len(temp_element) == 0:
            # print("start", current_height)
            temp_element.append(el)
            current_height = el[3]
            # continue
        elif i == len(element)-1:
            # print("end", current_height)
            temp_element.append(el)
            list_element.append(temp_element)
        elif current_height < ignore_threshold and el[3] < ignore_threshold:
            # print("small take", current_height)
            temp_element.append(el)
            current_height = (len(temp_element)*current_height + el[3])/(len(temp_element)+1)
            # continue
        elif el[3] / current_height > scale and len(temp_element)>1:
            # print("big sep", current_height)
            list_element.append(temp_element)
            temp_element = list()
            temp_element.append(el)
            current_height = el[3]
        elif current_height/el[3] > scale and current_height/element[i+1][3] > scale:
            # print("small sep", current_height)
            list_element.append(temp_element)
            temp_element = list()
            temp_element.append(el)
            current_height = el[3]
        else:
            temp_element.append(el)
    return np.array(list_element)


def heat_spliter(segmap, k):
    max_length = 10
    break_chars = 5
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(segmap.astype(np.uint8),
                                                                          connectivity=4)
    # print(k, n_labels, segmap.shape)
    element = list()
    x_coor = list()
    for k1 in range(1, n_labels):
        x, y = stats[k1, cv2.CC_STAT_LEFT], stats[k1, cv2.CC_STAT_TOP]
        w, h = stats[k1, cv2.CC_STAT_WIDTH], stats[k1, cv2.CC_STAT_HEIGHT]
        element.append([x, y, w, h])
        x_coor.append(x)
    element = np.array(element)
    idx = np.argsort(x_coor)
    # print(idx)
    element = element[idx]

    split_height = True
    if split_height:
        elements = height_spliter(element)
    else:
        elements = [element]

    split_length = True
    count = 0
    sep_height_seg = list()
    if split_length:
        temp_elements = list()
        for ele in elements:
            if len(ele) > max_length:
                temp_ele = [ele[x:x+break_chars] for x in range(0, len(ele), break_chars)]
                temp_elements.extend(temp_ele)
            else:
                temp_elements.append(ele)
        elements = temp_elements

    for j, ele in enumerate(elements):
        segmap_temp = np.zeros(segmap.shape, dtype=np.uint8)
        for e in ele:
            segmap_temp[labels == idx[count] + 1] = 255
            count += 1
        sep_height_seg.append(segmap_temp)
        # cv2.imwrite("data/vis/" + str(k) + "sep"+ str(j) +".jpg", segmap_temp)

    return sep_height_seg


def get_det_boxes_core(text_map, link_map, text_threshold, link_threshold, low_text):
    # prepare data
    img_h, img_w = text_map.shape

    # labeling method
    ret, text_score = cv2.threshold(text_map, low_text, 1, 0)
    ret, link_score = cv2.threshold(link_map, link_threshold, 1, 0)
    text_score_comb = np.clip(text_score + link_score, 0, 1)
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(text_score_comb.astype(np.uint8),
                                                                          connectivity=4)
    det = []
    mapper = []
    for k in range(1, n_labels):
        # size filtering
        size = stats[k, cv2.CC_STAT_AREA]
        if size < 10:
            continue

        # thresholding
        if np.max(text_map[labels == k]) < text_threshold:
            continue

        # make segmentation map
        segmap = np.zeros(text_map.shape, dtype=np.uint8)
        segmap[labels == k] = 255

        segmap[np.logical_and(link_score == 1, text_score == 0)] = 0  # remove link area
        x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]

        niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)
        sx, ex, sy, ey = x - niter, x + w + niter + 1, y - niter, y + h + niter + 1
        # boundary check
        if sx < 0:
            sx = 0
        if sy < 0:
            sy = 0
        if ex >= img_w:
            ex = img_w
        if ey >= img_h:
            ey = img_h
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1 + niter, 1 + niter))

        # cv2.imwrite("data/vis/" + str(k) + "crop.jpg", segmap[sy:ey, sx:ex])

        sep_seg = heat_spliter(segmap[sy:ey, sx:ex], k)
        for seg in sep_seg:
            temp_seg = segmap.copy()
            temp_seg[sy:ey, sx:ex] = cv2.dilate(seg, kernel, iterations=1)
            # make box
            np_contours = np.roll(np.array(np.where(temp_seg != 0)), 1, axis=0).transpose().reshape(-1, 2)
            # print(np_contours.shape)
            # temp_seg = cv2.cvtColor(segmap,cv2.COLOR_GRAY2RGB)
            # image = cv2.drawContours(temp_seg, [np_contours.astype(int)], -1, (0, 255, 0), 2)
            # cv2.imwrite("data/vis/" + str(k) + "contour.jpg", image)

            rectangle = cv2.minAreaRect(np_contours)
            box = cv2.boxPoints(rectangle)
            # print(box)
            # align diamond-shape
            w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
            box_ratio = max(w, h) / (min(w, h) + 1e-5)
            if abs(1 - box_ratio) <= 0.1:
                l, r = min(np_contours[:, 0]), max(np_contours[:, 0])
                t, b = min(np_contours[:, 1]), max(np_contours[:, 1])
                box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)

            # make clock-wise order
            start_idx = box.sum(axis=1).argmin()
            box = np.roll(box, 4 - start_idx, 0)
            box = np.array(box)

            det.append(box)
            mapper.append(k)

    return det, labels, mapper


def get_poly_core(boxes, labels, mapper, link_map):
    # configs
    num_cp = 5
    max_len_ratio = 0.7
    expand_ratio = 1.45
    max_r = 2.0
    step_r = 0.2

    polys = []
    for k, box in enumerate(boxes):
        # size filter for small instance
        w, h = int(np.linalg.norm(box[0] - box[1]) + 1), int(np.linalg.norm(box[1] - box[2]) + 1)
        if w < 30 or h < 30:
            polys.append(None)
            continue

        # warp image
        tar = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        M = cv2.getPerspectiveTransform(box, tar)
        word_label = cv2.warpPerspective(labels, M, (w, h), flags=cv2.INTER_NEAREST)
        try:
            min_v = np.linalg.inv(M)
        except:
            polys.append(None)
            continue

        # binarization for selected label
        cur_label = mapper[k]
        word_label[word_label != cur_label] = 0
        word_label[word_label > 0] = 1

        # Polygon generation
        # find top/bottom contours
        cp = []
        max_len = -1
        for i in range(w):
            region = np.where(word_label[:, i] != 0)[0]
            if len(region) < 2:
                continue
            cp.append((i, region[0], region[-1]))
            length = region[-1] - region[0] + 1
            if length > max_len:
                max_len = length

        # pass if max_len is similar to h
        if h * max_len_ratio < max_len:
            polys.append(None)
            continue

        # get pivot points with fixed length
        tot_seg = num_cp * 2 + 1
        seg_w = w / tot_seg  # segment width
        pp = [None] * num_cp  # init pivot points
        cp_section = [[0, 0]] * tot_seg
        seg_height = [0] * num_cp
        seg_num = 0
        num_sec = 0
        prev_h = -1
        for i in range(0, len(cp)):
            (x, sy, ey) = cp[i]
            if (seg_num + 1) * seg_w <= x and seg_num <= tot_seg:
                # average previous segment
                if num_sec == 0:
                    break
                cp_section[seg_num] = [cp_section[seg_num][0] / num_sec, cp_section[seg_num][1] / num_sec]
                num_sec = 0

                # reset variables
                seg_num += 1
                prev_h = -1

            # accumulate center points
            cy = (sy + ey) * 0.5
            cur_h = ey - sy + 1
            cp_section[seg_num] = [cp_section[seg_num][0] + x, cp_section[seg_num][1] + cy]
            num_sec += 1

            if seg_num % 2 == 0:
                continue  # No polygon area

            if prev_h < cur_h:
                pp[int((seg_num - 1) / 2)] = (x, cy)
                seg_height[int((seg_num - 1) / 2)] = cur_h
                prev_h = cur_h

        # processing last segment
        if num_sec != 0:
            cp_section[-1] = [cp_section[-1][0] / num_sec, cp_section[-1][1] / num_sec]

        # pass if num of pivots is not sufficient or segment width is smaller than character height
        if None in pp or seg_w < np.max(seg_height) * 0.25:
            polys.append(None)
            continue

        # calc median maximum of pivot points
        half_char_h = np.median(seg_height) * expand_ratio / 2

        # calc gradient and apply to make horizontal pivots
        new_pp = []
        for i, (x, cy) in enumerate(pp):
            dx = cp_section[i * 2 + 2][0] - cp_section[i * 2][0]
            dy = cp_section[i * 2 + 2][1] - cp_section[i * 2][1]
            if dx == 0:  # gradient if zero
                new_pp.append([x, cy - half_char_h, x, cy + half_char_h])
                continue
            rad = - math.atan2(dy, dx)
            c, s = half_char_h * math.cos(rad), half_char_h * math.sin(rad)
            new_pp.append([x - s, cy - c, x + s, cy + c])

        # get edge points to cover character heatmaps
        is_spp_found, is_epp_found = False, False
        grad_s = (pp[1][1] - pp[0][1]) / (pp[1][0] - pp[0][0]) + (pp[2][1] - pp[1][1]) / (pp[2][0] - pp[1][0])
        grad_e = (pp[-2][1] - pp[-1][1]) / (pp[-2][0] - pp[-1][0]) + (pp[-3][1] - pp[-2][1]) / (pp[-3][0] - pp[-2][0])
        for r in np.arange(0.5, max_r, step_r):
            dx = 2 * half_char_h * r
            if not is_spp_found:
                line_img = np.zeros(word_label.shape, dtype=np.uint8)
                dy = grad_s * dx
                p = np.array(new_pp[0]) - np.array([dx, dy, dx, dy])
                cv2.line(line_img, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), 1, thickness=1)
                if np.sum(np.logical_and(word_label, line_img)) == 0 or r + 2 * step_r >= max_r:
                    spp = p
                    is_spp_found = True
            if not is_epp_found:
                line_img = np.zeros(word_label.shape, dtype=np.uint8)
                dy = grad_e * dx
                p = np.array(new_pp[-1]) + np.array([dx, dy, dx, dy])
                cv2.line(line_img, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), 1, thickness=1)
                if np.sum(np.logical_and(word_label, line_img)) == 0 or r + 2 * step_r >= max_r:
                    epp = p
                    is_epp_found = True
            if is_spp_found and is_epp_found:
                break

        # pass if boundary of polygon is not found
        if not (is_spp_found and is_epp_found):
            polys.append(None)
            continue

        # make final polygon
        poly = [warp_coord(min_v, (spp[0], spp[1]))]
        for p in new_pp:
            poly.append(warp_coord(min_v, (p[0], p[1])))
        poly.append(warp_coord(min_v, (epp[0], epp[1])))
        poly.append(warp_coord(min_v, (epp[2], epp[3])))
        for p in reversed(new_pp):
            poly.append(warp_coord(min_v, (p[2], p[3])))
        poly.append(warp_coord(min_v, (spp[2], spp[3])))

        # add to final result
        polys.append(np.array(poly))

    return polys


def get_det_boxes(text_map, link_map, text_threshold, link_threshold, low_text, poly=False):
    boxes, labels, mapper = get_det_boxes_core(text_map, link_map, text_threshold, link_threshold, low_text)

    if poly:
        polys = get_poly_core(boxes, labels, mapper, link_map)
    else:
        polys = [None] * len(boxes)

    return boxes, polys


def adjust_result_coordinates(polys, ratio_w, ratio_h, ratio_net=2):
    if len(polys) > 0:
        polys = np.array(polys)
        for k in range(len(polys)):
            if polys[k] is not None:
                polys[k] *= (ratio_w * ratio_net, ratio_h * ratio_net)
    return polys
