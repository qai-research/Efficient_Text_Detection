import os
from pathlib import Path
import cv2
import numpy as np
import json
import argparse
from tqdm import tqdm
import random
import lmdb
import itertools

def dist(p1, p2):
    """Calculate the distance between 2 points

    Parameters
    ----------
    p1 : numpy array
        point 1, [x, y]
    p2 : numpy array
        point 2, [x, y]

    Returns
    ------
    float
        the distance
    """
    return np.linalg.norm(p1 - p2)


def find_closest(p1, box):
    """Find the closest point of a box to a given point

    Parameters
    ----------
    p1 : numpy array
        a point, [x, y]
    box : numpy array
        a box which contains 4 points, shape = (4, 2)

    Returns
    ------
    numpy array
        the closest point
    """
    closest = box[0]
    min_dist = dist(p1, box[0])
    for i in range(1, 4):
        d = dist(p1, box[i])
        if d < min_dist:
            min_dist = d
            closest = box[i]
    return closest


def check_valid_coors(coords):
    """Check if the given coords of the polygon are valid

    Parameters
    ----------
    coords : numpy array
        a list of coordinates, shape = (8,)

    Returns
    ------
    bool
        whether all coordinates are valid or not
    """
    for i in coords:
        if i < 0:
            return False
    return True


def crop(coords, image):
    """Crop and transform the zone's image within "image"

    Parameters
    ----------
    coords : numpy array
        a list of coordinates, shape = (8,)
    image : numpy array
        the big image that contains the zone

    Returns
    ------
    numpy array
        the cropped and transformed zone's image
    """
    poly = np.reshape(coords, (4, 1, 2))
    rect = cv2.minAreaRect(poly)
    poly = poly.reshape((4, 2))

    box = cv2.boxPoints(rect)
    box = np.int0(box).reshape((4, 2))

    ordered_box = []
    for i in range(4):
        ordered_box.append(find_closest(poly[i], box))
    src_pts = np.array(ordered_box).astype("float32")

    width = int(dist(ordered_box[0], ordered_box[1]))
    height = int(dist(ordered_box[0], ordered_box[3]))

    # coordinate of the points in box points after the rectangle has been straightened
    dst_pts = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
                       dtype="float32")

    # the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # directly warp the rotated rectangle to get the straightened rectangle
    warped = cv2.warpPerspective(image, M, (width, height))

    return warped


def coor_random_expand(coor, expand=None):
    coor = np.array(coor)
    coor = coor.reshape(4, 2)
    xc = [x[0] for x in coor]
    yc = [y[1] for y in coor]
    space_list = list()
    for i, x in enumerate(coor):
        for j, y in enumerate(coor):
            if i == j:
                continue
            vn = np.linalg.norm(x - y)
            space_list.append(vn)
    space = min(space_list)
    idx = np.argsort(xc)
    idy = np.argsort(yc)
    coor[idx[0]][0] -= random.randint(0, int(expand * space))
    coor[idx[1]][0] -= random.randint(0, int(expand * space))
    coor[idx[2]][0] += random.randint(0, int(expand * space))
    coor[idx[3]][0] += random.randint(0, int(expand * space))
    coor[idy[0]][1] -= random.randint(0, int(expand * space))
    coor[idy[1]][1] -= random.randint(0, int(expand * space))
    coor[idy[2]][1] += random.randint(0, int(expand * space))
    coor[idy[3]][1] += random.randint(0, int(expand * space))
    return coor.reshape(1, 8)


def write_cache(env, cache):
    """Write data to the database

    Parameters
    ----------
    env : Environment
        the lmdb Environment, which is the database.
    cache : dict
        the data to be written to the database
    """
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


def process_data(input_folder, name, output_folder, characters=None, expand=None,
                 db_map_size=50073741824, todetec=True, torecog=True):
    """Converting SynthText data into raw training data.

    Parameters
    ----------
    image_folder : str
        path to the images
    gt_folder : str
        path to the ground truth files
    output_folder : str
        path to the output folder
    sub_folder : str
        sub folder name
    characters: list
        list of selected characters, example whose label does not contain any of these are discarded
    """
    # create folders if not existed
    output_folder_path = Path(output_folder)
    output_folder_path.mkdir(exist_ok=True)


    if todetec:
        output_folder_path = Path(output_folder)
        output_folder_path = output_folder_path.joinpath('ST')
        output_folder_path.mkdir(exist_ok=True)

        output_detec = 'ST_' + name
        detec_path = output_folder_path.joinpath(output_detec)
        detec_path.mkdir(exist_ok=True)
        env_detec = lmdb.open(str(detec_path), map_size=db_map_size)
        cache_detec = {}

    if torecog:
        output_folder_path = Path(output_folder)
        output_folder_path = output_folder_path.joinpath('CR')
        output_folder_path.mkdir(exist_ok=True)

        output_recog = 'CR_' + name
        recog_path = output_folder_path.joinpath(output_recog)
        recog_path.mkdir(exist_ok=True)
        env_recog = lmdb.open(str(recog_path), map_size=db_map_size)
        cache_recog = {}

    if not torecog and not todetec:
        raise ("ples set todetec or torecog = True")

    image_folder = os.path.join(input_folder, 'images')
    annot_folder = os.path.join(input_folder, 'annotations')

    image_folder_path = Path(image_folder)
    image_files = [x for x in image_folder_path.iterdir() if x.is_file()]

    gt_folder_path = Path(annot_folder)

    vocabs = set()
    count_detec = 1
    count_recog = 1
    for image_file in tqdm(image_files):
        file_name = image_file.stem

        try:
            # img = cv2.imdecode(np.fromfile(str(image_file), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            with open(str(image_file), 'rb') as f:
                image_bin = f.read()  # read image as binary
                image_buf = np.frombuffer(image_bin, dtype=np.uint8)
                img = cv2.imdecode(image_buf, cv2.IMREAD_GRAYSCALE)
                img_h, img_w = img.shape[0], img.shape[1]
                if img_h * img_w == 0:
                    print("image have 0 size dimention")
                    continue
        except:
            print("cant read file")
            continue
        if img is None:
            print("image None")
            continue
        gt_file = gt_folder_path.joinpath("{}.json".format(file_name))
        try:
            with open(gt_file, "r", encoding='utf-8-sig') as f:
                obj = json.load(f)
                words = obj.get("words", [])
        except:
            print("cant read annotation")
            continue
        if todetec:
            # _, encode_img = cv2.imencode('.jpg', img)


            image_key_detec = 'image-{:09d}'.format(count_detec).encode()
            label_key_detec = 'label-{:09d}'.format(count_detec).encode()
            # _, temp_img = cv2.imencode('.jpg', encode_img)
            json_string = json.dumps(obj)

            cache_detec[image_key_detec] = image_bin
            cache_detec[label_key_detec] = json_string.encode()
            if count_detec % 1000 == 0:
                write_cache(env_detec, cache_detec)
                cache_detec = {}
            count_detec += 1
        if not torecog:
            continue
        for i in range(len(words)):
            label = words[i]['text']

            # check label if character list is given
            if characters is not None:
                valid_label = False
                for c in label:
                    if c in characters:
                        valid_label = True
                        break
                if not valid_label:
                    continue

            coords = [words[i]['x1'], words[i]['y1'],
                      words[i]['x2'], words[i]['y2'],
                      words[i]['x3'], words[i]['y3'],
                      words[i]['x4'], words[i]['y4']]

            if not check_valid_coors(coords):
                continue

            vocabs.update(label)
            coords = coor_random_expand(coords, expand=expand)
            # exit()
            cropped = crop(coords, img)
            # print(cropped.shape)
            image_key_recog = 'image-{:09d}'.format(count_recog).encode()
            label_key_recog = 'label-{:09d}'.format(count_recog).encode()
            _, temp_img = cv2.imencode('.jpg', cropped)
            cache_recog[image_key_recog] = temp_img.tobytes()
            cache_recog[label_key_recog] = label.encode()
            if count_recog % 1000 == 0:
                write_cache(env_recog, cache_recog)
                cache_recog = {}
            count_recog += 1

    if torecog:
        num_samples_recog = count_recog - 1
        cache_recog['num-samples'.encode()] = str(num_samples_recog).encode()
        write_cache(env_recog, cache_recog)
        print('Created recognition dataset with {} samples'.format(num_samples_recog))

    if todetec:
        num_samples_detec = count_detec - 1
        cache_detec['num-samples'.encode()] = str(num_samples_detec).encode()
        write_cache(env_detec, cache_detec)
        print('Created detection dataset with {} samples'.format(num_samples_detec))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='path to the ground truth files')
    parser.add_argument('--output', required=True, help='path to the output folder')
    parser.add_argument('--name', default='TYPE_LANG_vx_x', help='sub folder name')
    parser.add_argument('--expand', default=0.35, help='random expand boxes')
    parser.add_argument('--character', type=str, help='character label', default=None)
    parser.add_argument('--map_size', type=int, default=50073741824, help='maximum size database may grow to')

    opt = parser.parse_args()

    if opt.character is not None:
        opt.character = list(opt.character)

    process_data(input_folder=opt.input, name=opt.name,
                 output_folder=opt.output, characters=opt.character,
                 expand=opt.expand, db_map_size=opt.map_size)
