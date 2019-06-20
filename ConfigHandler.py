import configparser


class Config:

    def __init__(self,path):
        self.path = path

    def get(self, key):
        config = configparser.ConfigParser()
        config.read(self.path)

        try:
            value = config['DEFAULT'][key]
            return value

        except Exception:
            return None

