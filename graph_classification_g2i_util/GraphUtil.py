from graph_classification_g2i_config.ConfigHandler import Config
import networkx as nx
import logging


class GraphUtil:

    def get_graph_from_matrix(self, n):

        try:
            g = nx.from_numpy_matrix(n)
            return g
        except Exception as e:
            print('[Log]: Cant create graph from matrix')
            logging.exception(e)
            return None

    def get_filtered_matrix_fmri(self, n, config):

        # set sparsity to 24
        # binarization

        num_of_max_edges = (len(n) * (len(n) - 1)) / 2
        threshold = 0

        for i in range(0, 100):
            remaining_num = (n > threshold).sum()
            sparsity = (remaining_num / num_of_max_edges) * 100
            if abs(sparsity - int(config.get('fmri_sparsity_threshold'))) < 0.5:
                break
            threshold = threshold + 0.01
        n = (n > threshold).astype(int)

        return self.get_graph_from_matrix(n)
