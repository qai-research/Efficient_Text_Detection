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

