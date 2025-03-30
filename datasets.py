import os, sys
import scipy.io as sio
from util import *

def load_data(config):
    """Load data """
    data_name = config['dataset']
    main_dir = sys.path[0]
    X_list = []
    Y_list = []
    if data_name in ['Reuters_dim10']:
        mat = sio.loadmat(os.path.join(main_dir, 'data', data_name + '.mat'))
        #x0 = mat['x_train'][0]      # [9379, 10]
        #x1 = mat['x_train'][1]      # [9379, 10]
        X_list.append(normalize(np.vstack((mat['x_train'][0], mat['x_test'][0]))))
        X_list.append(normalize(np.vstack((mat['x_train'][1], mat['x_test'][1]))))
        Y_list.append(np.squeeze(np.hstack((mat['y_train'], mat['y_test']))))
    elif data_name in ['CUB']:
        mat = sio.loadmat(os.path.join(main_dir, 'data', 'CUB.mat'))
        X = mat['X'][0]

        x1 = X[1]   #[600, 300]
        x2 = X[0]   #[600, 1024]
        xx1 = np.copy(x1)
        xx2 = np.copy(x2)
        Y = np.copy(mat['gt'])
        index = [i for i in range(600)]
        np.random.seed(600)
        np.random.shuffle(index)
        for i in range(600):
            xx1[i] = x1[index[i]]
            xx2[i] = x2[index[i]]
            Y[i] = mat['gt'][index[i]]

        from sklearn import preprocessing
        min_max_scaler = preprocessing.MinMaxScaler()
        xx1 = min_max_scaler.fit_transform(xx1)
        xx2 = min_max_scaler.fit_transform(xx2)

        X_list.append(xx1)
        X_list.append(xx2)
        y = np.squeeze(Y).astype('int')
        Y_list.append(y)
        print(y)

    return X_list, Y_list


