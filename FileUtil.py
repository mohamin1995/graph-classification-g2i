import numpy as np
import logging


class FileUtil:

    def __init__(self, path):
        self.path = path

    def get_brain_matrix_from_file(self):
        try:
            n = np.loadtxt(self.path)
            return n
        except Exception as e:
            print('[Log]: Cant find file or cant read file')
            logging.exception(e)
            return None


