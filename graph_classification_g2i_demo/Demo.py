import sys, getopt
from graph_classification_g2i_util.FileUtil import *
from graph_classification_g2i_util.GraphUtil import *
from graph_classification_g2i_config.ConfigHandler import Config
from graph_classification_g2i_classificaion.ClassificationUtil import ClassificationUtil
from sklearn.model_selection import  LeaveOneOut

import numpy as np


argv = '-c D:/conf.ini'.split()

def main(args):
    try:
        pts, args = getopt.getopt(argv, 'c:')
        config = Config(pts[0][1])

        fmri_asd_path = config.get('fmri_asd_path')
        fmri_td_path = config.get('fmri_td_path')
        brain_net_file_format = config.get('brain_net_file_format')

        g_util = GraphUtil()

        X = []
        Y = []


        for k in range(1, 41):
            n = FileUtil(fmri_asd_path + str(k) + brain_net_file_format).get_brain_matrix_from_file()
            G = g_util.get_filtered_matrix_fmri(n,config)

            sortedDic = sorted(G.degree, key=lambda x: x[1], reverse=True)

            m = len(sortedDic)
            newAdjMatrix = np.zeros((m, m), np.uint8)

            for i in range(0, m):
                for j in range(0, m):
                    if (G.has_edge(sortedDic[i][0], sortedDic[j][0])):
                        newAdjMatrix[i][j] = 255
                    else:
                        newAdjMatrix[i][j] = 0

            #cv2.imshow('im', newAdjMatrix)
            #cv2.waitKey()

            X.append(newAdjMatrix.flatten())
            Y.append(1)
            print('[' + str(k) + '] ASD subject processed successfully...')


        for k in range(1, 38):
            n = FileUtil(fmri_td_path + str(k) + brain_net_file_format).get_brain_matrix_from_file()
            G = g_util.get_filtered_matrix_fmri(n,config)

            sortedDic = sorted(G.degree, key=lambda x: x[1], reverse=True)

            m = len(sortedDic)
            newAdjMatrix = np.zeros((m, m), np.uint8)

            for i in range(0, m):
                for j in range(0, m):
                    if (G.has_edge(sortedDic[i][0], sortedDic[j][0])):
                        newAdjMatrix[i][j] = 255
                    else:
                        newAdjMatrix[i][j] = 0

            #cv2.imshow('im', newAdjMatrix)
            #cv2.waitKey()

            X.append(newAdjMatrix.flatten())
            Y.append(0)
            print('[' + str(k) + '] TD subject processed successfully...')



        X_array = np.array(X)
        y_array = np.array(Y)



        loo = LeaveOneOut()
        ytests = []
        ypreds = []

        for train_idx, test_idx in loo.split(X):
            X_train, X_test = X_array[train_idx], X_array[test_idx]  # requires arrays
            y_train, y_test = y_array[train_idx], y_array[test_idx]
            model =  ClassificationUtil.get_dt_model(X_train,y_train)
            y_pred = ClassificationUtil.classify(model,X_test)
            ytests += list(y_test)
            ypreds += list(y_pred)

        acc = ClassificationUtil.get_accuracy(ytests, ypreds)
        prec = ClassificationUtil.get_precision(ytests, ypreds)
        rec = ClassificationUtil.get_recall(ytests, ypreds)


        print('accuracy = '+str(acc))
        print('precision = '+str(prec))
        print('recall = '+str(rec))


    except getopt.GetoptError:
           print('Something went wrong!')
           sys.exit(2)

if __name__ == "__main__":
   main(sys.argv[1:])