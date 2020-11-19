import os
import glob
import sys
import random
import json
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw, ImageFont

class TextFontGenerator():
    
    def __init__(self, fonts_path, font_size_range, fixed_box = True, random_color = False, font_color = (0,0,0)):
        self.fonts_list = glob.glob(os.path.join(fonts_path,"*.ttf"))
        self.fonts_list.extend(glob.glob(os.path.join(fonts_path,"*.TTF")))
        self.font_size_range = font_size_range
        self.fixed_box = fixed_box
        self.random_color = random_color
        self.font_color = font_color
            
    def generator(self,source_word):
        font_path = random.choice(self.fonts_list)

        if self.font_size_range is not None:
            l,h = self.font_size_range
            font_size = np.random.randint(low = l, high = h)
        else:
            font_size = np.random.randint(10,50)

        if self.fixed_box is True:
            img,out_json = self.fixed_box_gen(source_word, font_path, font_size)
        else:
            img,out_json  = self.none_fixed_box_gen(source_word, font_path, font_size)

        return img,out_json 

    def fixed_box_gen(self,word,font_path,font_size):

        # Make full images
        if self.random_color is False:            
            color = self.font_color
        else:
            color = (np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255))

        fnt = ImageFont.truetype(font_path, font_size)
        full_img = Image.new('RGB', (font_size*(len(word)+2), font_size*4), color = (255, 255, 255))
        full_draw = ImageDraw.Draw(full_img)
        full_draw.text((font_size,font_size),word, font=fnt, fill=color)

        all_black = np.argwhere(np.array(full_img)[:,:,:]!=255)

        global_min_y = min(all_black[:,0])-1
        global_max_y = max(all_black[:,0])+1
        global_min_x = min(all_black[:,1])-1
        global_max_x = max(all_black[:,1])+1
        
        full_img = full_img.crop([global_min_x,global_min_y,global_max_x,global_max_y])
        full_box = [(global_min_x-global_min_x,global_min_y-global_min_y),(global_max_x-global_min_x,global_max_y-global_min_y)]

        # Determine charactor box
        old_img = Image.new('RGB', (font_size*(len(word)+2), font_size*4), color = color)
        out_json = {"words":word,
                    "text":[]}
        for i,w in enumerate(word):
            try:
                char = {'char':w}
                new_img = Image.new('RGB', (font_size*(len(word)+2), font_size*4), color = color)
                d = ImageDraw.Draw(new_img)
                d.text((font_size,font_size),word[:i+1], font=fnt, fill=(255, 255, 255))
                char_img = 255 - (np.array(new_img) - np.array(old_img))
                old_img = new_img.copy()

                all_black = np.argwhere(np.array(char_img)[:,:,:]!=255)
                if word[i] != ' ':
                    min_y = int(min(all_black[:,0]) - global_min_y)-1
                    max_y = int(max(all_black[:,0]) - global_min_y)+1
                    min_x = int(min(all_black[:,1]) - global_min_x)-1
                    max_x = int(max(all_black[:,1]) - global_min_x)+1
                    char['x1'] = min_x
                    char['y1'] = min_y
                    char['x2'] = max_x
                    char['y2'] = min_y
                    char['x3'] = max_x
                    char['y3'] = max_y
                    char['x4'] = min_x
                    char['y4'] = max_y
                    out_json['text'].append(char)
            except ValueError:
                pass
        return full_img, out_json
        

    def none_fixed_box_gen(self,word,font_path,font_size):

        # Make full images
        if self.random_color is False:            
            color = self.font_color
        else:
            color = (np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255))

        fnt = ImageFont.truetype(font_path, font_size)
        full_img = Image.new('RGB', (font_size*(len(word)+2), font_size*4), color = (255, 255, 255))
        full_draw = ImageDraw.Draw(full_img)
        full_draw.text((font_size,font_size),word, font=fnt, spacing = 100, fill=color)

        all_black = np.argwhere(np.array(full_img)[:,:,:]!=255)

        global_min_y = min(all_black[:,0])-1
        global_max_y = max(all_black[:,0])+1
        global_min_x = min(all_black[:,1])-1
        global_max_x = max(all_black[:,1])+1
        
        full_img = full_img.crop([global_min_x,global_min_y,global_max_x,global_max_y])
        full_box = [(global_min_x-global_min_x,global_min_y-global_min_y),(global_max_x-global_min_x,global_max_y-global_min_y)]

        # Determine charactor box
        old_img = Image.new('RGB', (font_size*(len(word)+2), font_size*4), color = color)
        out_json = {"words":word,
                    "text":[]}
        for i,w in enumerate(word):
            char = {'char':w}
            new_img = Image.new('RGB', (font_size*(len(word)+2), font_size*4), color = color)
            d = ImageDraw.Draw(new_img)
            d.text((font_size,font_size),word[:i+1], font=fnt, fill=(255, 255, 255))
            char_img = 255 - (np.array(new_img) - np.array(old_img))
            old_img = new_img.copy()

            all_black = np.argwhere(np.array(char_img)[:,:,:]!=255)
            if word[i] != ' ':
                min_y = 0
                max_y = global_max_y
                min_x = int(min(all_black[:,1]) - global_min_x)
                max_x = int(max(all_black[:,1]) - global_min_x)
                char['x1'] = min_x
                char['y1'] = min_y
                char['x2'] = max_x
                char['y2'] = min_y
                char['x3'] = max_x
                char['y3'] = max_y
                char['x4'] = min_x
                char['y4'] = max_y
                out_json['text'].append(char)

        return full_img, out_json