
from graph_classification_g2i_config.ConfigHandler import Config


conf_reader = Config('D:/conf.ini')
print(conf_reader.get('name'))


