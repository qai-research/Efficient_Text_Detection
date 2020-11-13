from pathlib import Path
import configparser


class Constants:
    def __init__(self, config_path):
        config_p = Path(config_path)
        if not config_p.is_file():
            raise ValueError("Config file not found")
        config_parser = configparser.ConfigParser()
        config_parser.read(config_path)
        self.config = config_parser['DEFAULT']
        print('loaded config from : ', str(config_p.resolve()))


def read_vocab(file_path):
    with open(file_path, "r", encoding='utf-8-sig') as f:
        list_lines = f.read().strip().split("\n")
    return list_lines


def read_json_annotation(js_label):
    js_words = js_label["words"]
    character_boxes = []
    words = []
    for tex in js_words:  # loop through each word
        bboxes = []
        words.append(tex['text'])
        for ch in tex['chars']:
            bo = [ch['x1'], ch['y1']], [ch['x2'], ch['y2']], [ch['x3'], ch['y3']], [ch['x4'], ch['y4']]
            bo = np.array(bo)
            bboxes.append(bo)
        character_boxes.append(np.array(bboxes))
    return words, character_boxes
