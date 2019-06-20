import sys
import getopt
from FileUtil import *
from GraphUtil import *
from ConfigHandler import Config
from ClassificationUtil import ClassificationUtil
from sklearn.model_selection import LeaveOneOut

import numpy as np


def main(argv):
    try:
        pts, args = getopt.getopt(argv, 'c:')

        config = Config(pts[0][1])

        fmri_asd_path = config.get('fmri_asd_path')
        fmri_td_path = config.get('fmri_td_path')
        brain_net_file_format = config.get('brain_net_file_format')

        g_util = GraphUtil()
        classification_util = ClassificationUtil()

        f_vector = []
        label = []

        for k in range(1, 41):
            n = FileUtil(fmri_asd_path + str(k) + brain_net_file_format).get_brain_matrix_from_file()
            G = g_util.get_filtered_matrix_fmri(n,config)

            sorted_dic = sorted(G.degree, key=lambda x: x[1], reverse=True)

            m = len(sorted_dic)
            new_adj_matrix = np.zeros((m, m), np.uint8)

            for i in range(0, m):
                for j in range(0, m):
                    if G.has_edge(sorted_dic[i][0], sorted_dic[j][0]):
                        new_adj_matrix[i][j] = 255
                    else:
                        new_adj_matrix[i][j] = 0

            '''cv2.imshow('im', newAdjMatrix)
            cv2.waitKey()'''

            f_vector.append(new_adj_matrix.flatten())
            label.append(1)
            print('[' + str(k) + '] ASD subject processed successfully...')

        for k in range(1, 38):
            n = FileUtil(fmri_td_path + str(k) + brain_net_file_format).get_brain_matrix_from_file()
            G = g_util.get_filtered_matrix_fmri(n,config)

            sorted_dic = sorted(G.degree, key=lambda x: x[1], reverse=True)

            m = len(sorted_dic)
            new_adj_matrix = np.zeros((m, m), np.uint8)

            for i in range(0, m):
                for j in range(0, m):
                    if G.has_edge(sorted_dic[i][0], sorted_dic[j][0]):
                        new_adj_matrix[i][j] = 255
                    else:
                        new_adj_matrix[i][j] = 0

            '''cv2.imshow('im', newAdjMatrix)
            cv2.waitKey()'''

            f_vector.append(new_adj_matrix.flatten())
            label.append(0)
            print('[' + str(k) + '] TD subject processed successfully...')

        x_array = np.array(f_vector)
        y_array = np.array(label)

        loo = LeaveOneOut()
        ytests = []
        ypreds = []

        for train_idx, test_idx in loo.split(x_array):
            x_train, x_test = x_array[train_idx], x_array[test_idx]  # requires arrays
            y_train, y_test = y_array[train_idx], y_array[test_idx]
            model = classification_util.get_dt_model(x_train, y_train)
            y_pred = classification_util.classify(model, x_test)
            ytests += list(y_test)
            ypreds += list(y_pred)

        acc = classification_util.get_accuracy(ytests, ypreds)
        prec = classification_util.get_precision(ytests, ypreds)
        rec = classification_util.get_recall(ytests, ypreds)

        print('[log]: accuracy = '+str(acc))
        print('[log]: precision = '+str(prec))
        print('[log]: recall = '+str(rec))

    except getopt.GetoptError:
        print('[error]: Invalid Input Args')
        sys.exit(2)


if __name__ == "__main__":
   main(sys.argv[1:])