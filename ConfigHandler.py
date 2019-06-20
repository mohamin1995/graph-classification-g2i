import configparser


class Config:

    def __init__(self, path):
        self.path = path
        self.conf = configparser.ConfigParser()
        self.conf.read(self.path)

    def get(self, key):

        try:
            value = self.conf['DEFAULT'][key]
            return value

        except Exception:
            return None

