import os
import sys
import cv2
import json
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from akaocr.SynthText.ImgtextProcessing.PerspectiveTransformation import Transform
from akaocr.SynthText.TextToImage.fromfont import TextFontGenerator
from akaocr.SynthText.BackgroundProcessing.box_generator import BoxGenerator
from akaocr.utils.text_gen import TextGenerator

def generate(fonts_path, font_size_range, target_json_path, target_image, random_color = False, font_color = (0,0,0),
             new_text_gen = False, fixed_box = True, 
             output_path = None, vocab_path = None, method = 'white', **kwargs):

    if method=='white' is False:
        new_text_gen = True


    # Load target label
    try:
        with open(target_json_path,'r',encoding = 'utf-8-sig') as reader:
            target_json = json.loads(reader.read())
    except:
        target_json = target_json_path

    target_points = [np.float32([[(char['x1'],char['y1']),
                                (char['x2'],char['y2']),
                                (char['x3'],char['y3']),
                                (char['x4'],char['y4'])]]) for char in target_json['words']]
    # Create new text generator 
    if new_text_gen:
        try:
            source_text_path = kwargs.pop('source_text_path')
        except KeyError:
            raise KeyError("The source_text_path (str) is required.")
            
        try:
            text_gen = TextGenerator(source_text_path,
                                     vocab_group_path = vocab_path, 
                                     min_length=kwargs.pop('min_length'), 
                                     max_length=kwargs.pop('max_length'),
                                     replace_percentage = 1)
        except KeyError:
            try:
                text_gen = TextGenerator(source_text_path,
                                        vocab_group_path = vocab_path, 
                                        min_length=kwargs.pop('min_length'),
                                        replace_percentage = 1)
            except KeyError:
                text_gen = TextGenerator(source_text_path,
                                        vocab_group_path = vocab_path, 
                                        replace_percentage = 1)
    
    # Create font to images generator
    font_to_img_gen = TextFontGenerator(fonts_path, font_size_range, 
                                        fixed_box = fixed_box,
                                        random_color = random_color, font_color = font_color)
    num_samples = len(target_points)


    # Generator new text to replace old text
    source_images= []
    source_chars_coor = []
    for word in target_json['words']:

        if new_text_gen:
            ## Generator text with out any infomation of source text
            img,out_json = font_to_img_gen.generator(text_gen.generate(opt_len = len(word["text"])))
        else:
            ## Generator text with infomation of source text
            template = word['text']
            vocab_dict = {}
            if vocab_path is not None:
                with open(vocab_path, "r", encoding='utf-8-sig') as f:
                    vocab_group = json.loads(f.read())
                for l in vocab_group:
                    for j in l:
                        if j!='':
                            vocab_dict[j]=l

            for i in vocab_dict:
                if i in template:
                    for _ in range(template.count(i)):
                        template = template.replace(i,random.choice(vocab_dict[i]), 1)
            img,out_json = font_to_img_gen.generator(template)

        source_images.append(np.array(img))
        source_chars_coor.append(out_json)

    # Preprocess the clean background
    clearned_target_image = cv2.imread(target_image)
    if method=='white':
        cv2.fillPoly(clearned_target_image, np.int32(target_points),[255,255,255])
        mask_img = np.uint8(np.zeros(clearned_target_image.shape))
        cv2.fillPoly(mask_img, np.int32(target_points),[255,255,255])
        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
        clearned_target_image = cv2.inpaint(clearned_target_image, mask_img, 3, flags=cv2.INPAINT_NS)

    # Add text images to background
    trans = Transform(source_images, source_chars_coor, target_points, target_image)
    result_pic, out_json_trans =  trans.fit()

    # Save images and json
    if output_path is not None:
        image_path = os.path.join(output_path,'images/%s.png'%target_json['file'][:-4])
        json_path = os.path.join(output_path,'annotations/%s.json'%target_json['file'][:-5])
        cv2.imwrite(image_path)
        with open(json_path,'w',encoding = 'utf-8-sig') as fw:
            fw.write(json.dumps(out_json_trans))
    return result_pic, out_json_trans