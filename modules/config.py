import yaml


class Config:
    
    def __init__(self, filename: str = "config.yaml"):
        with open(filename) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)