import os
import io
import sys
import time
import config
import argparse
import pandas as pd
import streamlit as st


sys.path.append('../akaocr')
from SynthText.SynthText_v2_1 import BlackList
from SynthText.SynthText_v2_1 import WhiteList

def main():
    st.markdown("<h1 style='text-align: center; color: Blue;'>SynthText App</h1>", unsafe_allow_html=True)

    # CREATE BACKGROUND DATAFRAME
    existed_background      = sorted([os.path.join(config.background_folder,name) for name in os.listdir(config.background_folder)])
    whitelist_background    = [path for path in existed_background if os.path.isdir(path) and 'anotations' in os.listdir(path)]
    blacklist_background    = [path for path in existed_background if os.path.isdir(path)]
    bg_df = {"NAME"     :[],
             "METHOD"   :[],
             "SIZE"     :[],
             "PATH"     :[]
            }
    
    for path in existed_background:

        if not len(os.listdir(path))>0:
            continue

        if path in blacklist_background:
            bg_df['NAME'].append(os.path.basename(path))
            bg_df['METHOD'].append('black')
            bg_df['SIZE'].append(len(os.listdir(path+"/images")))
            bg_df['PATH'].append(path)

            
        if path in whitelist_background:
            bg_df['NAME'].append(os.path.basename(path))
            bg_df['METHOD'].append('white')
            bg_df['SIZE'].append(len(os.listdir(path+"/images")))
            bg_df['PATH'].append(path)
        
    st.markdown("<h3 style='text-align: left; color: Blue;'>Backgrounds List</h3>", unsafe_allow_html=True)
    bg_df = pd.DataFrame(bg_df, columns = ["NAME","METHOD","SIZE","PATH"])
    st.dataframe(bg_df)

    # CREATE SOURCE DATAFRAME
    existed_source   = sorted([os.path.join(config.source_folder,name) for name in os.listdir(config.source_folder)])
    source_df    = {"NAME"       :[],
                   "SIZE"       :[],
                   "PATH"     :[],
                   "TYPE"     :[]          
                  }
    for path in existed_source:
        if os.path.isfile(path) and path.endswith('.txt'):
            with open(path, 'r', encoding = 'utf8') as fr:
                length = len(fr.read().split("\n"))
            source_df["NAME"].append(os.path.basename(path))
            source_df["SIZE"].append(length)
            source_df["PATH"].append(path)
            source_df["TYPE"].append('Text type')

        elif os.path.isdir(path) and 'images' in os.listdir(path) and 'labels' in os.listdir(path):
            files = os.listdir(path)
            length = len(os.listdir(os.path.join(path,'images')))
            source_df["NAME"].append(os.path.basename(path))
            source_df["SIZE"].append(length)
            source_df["PATH"].append(path)
            source_df["TYPE"].append('Object type')
            
    st.markdown("<h3 style='text-align: left; color: Blue;'>Text Source List</h3>", unsafe_allow_html=True)
    source_df = pd.DataFrame(source_df, columns = ["NAME","TYPE","SIZE","PATH"])
    st.dataframe(source_df)

    # CREATE FONT DATAFRAME
    existed_font   = sorted([os.path.join(config.font_folder,name) for name in os.listdir(config.font_folder)])

    font_df    = {"NAME"       :[],
                   "SIZE"       :[],
                   "PATH"     :[]             
                  }
    for path in existed_font:
        if os.path.isdir(path):            
            font_df["NAME"].append(os.path.basename(path))
            font_df["SIZE"].append(len(os.listdir(path)))
            font_df["PATH"].append(path)
    st.markdown("<h3 style='text-align: left; color: Blue;'>Fonts List</h3>", unsafe_allow_html=True)
    font_df = pd.DataFrame(font_df, columns = ["NAME","SIZE","PATH"])
    st.dataframe(font_df)

    file_buffer = st.file_uploader("UPLOAD CONFIG FILES")
    if file_buffer is not None:
        config_file = pd.read_csv(file_buffer)
        key = config_file.columns
        checked_df  = check_valild(config_file, bg_df, source_df, font_df)
        key = key.insert(0,'DETAIL')
        key = key.insert(0,'STATUS')

        st.text("The uploaded config")
        st.dataframe(checked_df[key])
        if st.button("START GEN"):
            for index, value in enumerate(checked_df.values):

                Method,Fonts,Backgrounds,Textsources,num_images,max_num_box = value[:6]
                char_spacing,min_font_size,max_font_size,min_text_length,max_text_length,random_color,max_num_text = value[:-4][6:]
                max_height, max_width, status, detail = value[-4:]
                if status is "INVALID":
                    continue
                parser = argparse.ArgumentParser()
                opt = parser.parse_args()

                opt.method = Method
                opt.backgrounds_path = os.path.join(config.background_folder,Backgrounds,'images')

                opt.fonts_path = os.path.join(config.font_folder,Fonts)
                opt.font_size_range = (min_font_size, max_font_size)
                opt.fixed_box = True
                opt.num_images = num_images
                opt.output_path = os.path.join(config.outputs_folder,Backgrounds)
                opt.source_path = os.path.join(config.source_folder, Textsources)
                opt.random_color = (random_color==1)
                opt.font_color   = (0,0,0)
                opt.min_text_length = min_text_length
                opt.max_text_length = max_text_length
                opt.max_num_text = None
                opt.max_size = (max_height, max_width)

                st.warning("Begin running SynthText with %s folder"%Backgrounds)

                if opt.method == 'white':
                    a = time.time()
                    opt.input_json = os.path.join(config.background_folder,Backgrounds,'anotations')
                    runner = WhiteList(opt)

                elif opt.method == 'black':
                    a = time.time()
                    if Textsources in source_df[source_df['TYPE'] == 'Object type']['NAME'].values:
                        opt.is_object = True
                    opt.fixed_size = None
                    opt.weigh_random_range = (30,100)
                    opt.heigh_random_range = (10,50)
                    opt.box_iter = 100
                    opt.max_num_box = max_num_box
                    opt.num_images = num_images
                    opt.aug_percent = 0
                    seg_path = os.path.join(config.background_folder,Backgrounds,'seg.h5')
                    opt.segment = seg_path if os.path.exists(seg_path) else None
                    opt.segment = None
                    runner = BlackList(opt)

                elif opt.method == 'double_black':
                    a = time.time()
                    opt.method = 'black'
                    opt.fixed_size = None
                    opt.weigh_random_range = (30,100)
                    opt.heigh_random_range = (10,50)
                    opt.box_iter = 100
                    opt.max_num_box = max_num_box*10
                    opt.num_images = num_images
                    opt.aug_percent = 0
                    opt.segment = None
                    opt.source_path = '/home/vietvh9/Project/OCR_Components/data/sources/source.txt'
                    opt.is_object = False
                    runner = BlackList(opt, is_return = True)
                    new_backgrounds_path = runner.run()

                    
                    opt.source_path = os.path.join(config.source_folder, Textsources)
                    if Textsources in source_df[source_df['TYPE'] == 'Object type']['NAME'].values:
                        opt.is_object = True
                    opt.max_num_box = max_num_box
                    opt.backgrounds_path = os.path.join(new_backgrounds_path,'images')
                    opt.output_path = os.path.join(new_backgrounds_path,"Double_black")
                    runner = BlackList(opt, is_random_sample = False)
                runner.run()
                st.text("Time for this process was %s seconds"%(time.time() - a))

def check_valild(dataframe, bg_df, source_df, fonts_df):
    df = dataframe.copy()
    results = {}
    for index, value in enumerate(dataframe.values):
        results[index] = {"Status":"valid", "Error":[]}
        Method,Fonts,Backgrounds,Textsources = value[:4]
        if Backgrounds not in bg_df['NAME'].values:
            results[index]['Error'].append('The backgrounds does not existed.')
        else:
            info =  bg_df[bg_df['NAME']==Backgrounds]
            st.dataframe(info)
            if Method == 'whitelist' and info['METHOD'].values[0] != 'whitelist':
                results[index]['Error'].append('The backgrounds can not gen with whitelist method.')

            if Fonts not in fonts_df['NAME'].values:
                results[index]['Error'].append('The fonts does not existed.')

            if Textsources not in source_df['NAME'].values:
                results[index]['Error'].append('The text sources does not existed.')
        if len(results[index]['Error']) is not 0:
            results[index]["Status"]="INVALID"
    df['STATUS'] = [results[i]["Status"] for i in range(len(results))]
    df['DETAIL'] = [results[i]["Error"] for i in range(len(results))]
    return df


if __name__ == '__main__':    

    st.set_option('deprecation.showfileUploaderEncoding', False)
    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    local_css("style.css")
    main()