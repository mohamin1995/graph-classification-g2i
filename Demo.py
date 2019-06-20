import sys
import getopt
from FileUtil import *
from GraphUtil import *
from ConfigHandler import Config
from ClassificationUtil import ClassificationUtil
from sklearn.model_selection import LeaveOneOut
from os import listdir
from os.path import isfile, join

import numpy as np


def main(argv):
    try:
        pts, args = getopt.getopt(argv, 'c:')

        config = Config(pts[0][1])

        fmri_asd_path = config.get('fmri_asd_path')
        fmri_td_path = config.get('fmri_td_path')

        if fmri_asd_path is None or fmri_td_path is None:
            print('[error]: Config Not Found or Bad Config')
            return

        g_util = GraphUtil()
        classification_util = ClassificationUtil()

        f_vector = []
        label = []

        files = [f for f in listdir(fmri_asd_path) if isfile(join(fmri_asd_path, f))]

        if len(files) == 0:
            print('[error]: asd directory is empty')
            return

        for k in range(0, len(files)):
            n = FileUtil(fmri_asd_path + files[k]).get_brain_matrix_from_file()

            try:
                g = g_util.get_filtered_matrix_fmri(n, config)

            except Exception:
                print('[error]: Incorrect matrix structure')
                return

            sorted_dic = sorted(g.degree, key=lambda x: x[1], reverse=True)

            m = len(sorted_dic)
            new_adj_matrix = np.zeros((m, m), np.uint8)

            for i in range(0, m):
                for j in range(0, m):
                    if g.has_edge(sorted_dic[i][0], sorted_dic[j][0]):
                        new_adj_matrix[i][j] = 255
                    else:
                        new_adj_matrix[i][j] = 0

            '''cv2.imshow('im', new_adj_matrix)
            cv2.waitKey()'''

            f_vector.append(new_adj_matrix.flatten())
            label.append(1)
            print('[log]: ' + str(k+1) + ' ASD subject processed successfully...')

        files = [f for f in listdir(fmri_td_path) if isfile(join(fmri_td_path, f))]

        if len(files) == 0:
            print('[error]: td directory is empty')
            return

        for k in range(0, len(files)):
            n = FileUtil(fmri_td_path + files[k]).get_brain_matrix_from_file()

            try:
                g = g_util.get_filtered_matrix_fmri(n, config)

            except Exception:
                print('[error]: Incorrect matrix structure')
                return

            sorted_dic = sorted(g.degree, key=lambda x: x[1], reverse=True)

            m = len(sorted_dic)
            new_adj_matrix = np.zeros((m, m), np.uint8)

            for i in range(0, m):
                for j in range(0, m):
                    if g.has_edge(sorted_dic[i][0], sorted_dic[j][0]):
                        new_adj_matrix[i][j] = 255
                    else:
                        new_adj_matrix[i][j] = 0

            '''cv2.imshow('im', new_adj_matrix)
            cv2.waitKey()'''

            f_vector.append(new_adj_matrix.flatten())
            label.append(0)
            print('[log]: ' + str(k+1) + ' TD subject processed successfully...')

        print('[log]: Evaluating...')

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

        print('.......... [report] ..........')
        print('+ accuracy = '+str(acc))
        print('+ precision = '+str(prec))
        print('+ recall = '+str(rec))

    except getopt.GetoptError:
        print('[error]: Invalid Input Args')
        return


if __name__ == "__main__":
   main(sys.argv[1:])