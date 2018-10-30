import csv, itertools
import os
from os.path import join as os_join
import numpy as np
from sklearn import tree, svm, preprocessing
from sklearn.externals import joblib
from CICIDS2017 import read_data

dataroot = r'F:\arash\CIC-IDS-2017\MachineLearningCSV'

attacks = {'BENIGN': 0, 'Bot': 1, 'DDoS': 2, 'DoS GoldenEye': 3, 'DoS Hulk': 4, 'DoS Slowhttptest': 5, 'DoS slowloris': 6, 'FTP-Patator': 7, 'Heartbleed': 8, 'Infiltration': 9,\
 'PortScan': 10, 'SSH-Patator': 11, 'Web Attack ï¿½ Brute Force': 12, 'Web Attack ï¿½ Sql Injection': 13, 'Web Attack ï¿½ XSS': 14}

feature_names = [' Destination Port', ' Flow Duration', ' Total Fwd Packets', ' Total Backward Packets', 'Total Length of Fwd Packets', ' Total Length of Bwd Packets', ' Fwd Packet Length Max', \
' Fwd Packet Length Min', ' Fwd Packet Length Mean', ' Fwd Packet Length Std', 'Bwd Packet Length Max', ' Bwd Packet Length Min', 'Bwd Packet Length Mean', ' Bwd Packet Length Std', 'Flow Bytes/s',\
 ' Flow Packets/s', ' Flow IAT Mean', ' Flow IAT Std', ' Flow IAT Max', ' Flow IAT Min', 'Fwd IAT Total', ' Fwd IAT Mean', ' Fwd IAT Std', ' Fwd IAT Max', ' Fwd IAT Min', 'Bwd IAT Total', ' Bwd IAT Mean',\
  ' Bwd IAT Std', ' Bwd IAT Max', ' Bwd IAT Min', 'Fwd PSH Flags', ' Bwd PSH Flags', ' Fwd URG Flags', ' Bwd URG Flags', ' Fwd Header Length', ' Bwd Header Length', 'Fwd Packets/s', ' Bwd Packets/s',\
   ' Min Packet Length', ' Max Packet Length', ' Packet Length Mean', ' Packet Length Std', ' Packet Length Variance', 'FIN Flag Count', ' SYN Flag Count', ' RST Flag Count', ' PSH Flag Count', ' ACK Flag Count',\
    ' URG Flag Count', ' CWE Flag Count', ' ECE Flag Count', ' Down/Up Ratio', ' Average Packet Size', ' Avg Fwd Segment Size', ' Avg Bwd Segment Size', ' Fwd Header Length', 'Fwd Avg Bytes/Bulk', \
    ' Fwd Avg Packets/Bulk', ' Fwd Avg Bulk Rate', ' Bwd Avg Bytes/Bulk', ' Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate', 'Subflow Fwd Packets', ' Subflow Fwd Bytes', ' Subflow Bwd Packets', ' Subflow Bwd Bytes',\
     'Init_Win_bytes_forward', ' Init_Win_bytes_backward', ' act_data_pkt_fwd', ' min_seg_size_forward', 'Active Mean', ' Active Std', ' Active Max', ' Active Min', 'Idle Mean', ' Idle Std', ' Idle Max', ' Idle Min']
np.random.seed(1202)


def convert2npy(data):
    arr = np.array(data)
    X = arr[:,:-1].astype(np.float32)
    Y_str = arr[:,-1]
    rows_without_nan = ~np.isnan(X).any(axis=1)
    X = X[rows_without_nan]
    Y_str = Y_str[rows_without_nan]

    rows_with_finite = np.isfinite(X).all(axis=1)
    X = X[rows_with_finite]
    Y_str = Y_str[rows_with_finite]   
    
    Y = [attacks[y_str] for y_str  in Y_str]
    return (X,Y)


def sample_data(data,sampling_rate):
    data = np.random.permutation(data)
    N = len(data)
    idx = np.random.randint(N, size=N//sampling_rate)
    data = data[idx]
    return data

def split_data(data): # train,val,test -> 80%,5%,15%
    data = np.random.permutation(data)
    N = len(data)
    num_train = int(N*80/100)
    num_val = int(N*5/100)
    train_data, val_data, test_data = data[:num_train], data[num_train:num_train+num_val], data[num_train+num_val:]
    return (train_data, val_data, test_data)


data = read_data(dataroot)
print('data is read!')
data = sample_data(data,10)
print('data is sampled!')

print('There are #{} rows'.format(len(data)))
train_data, val_data, test_data = split_data(data)
X_train,Y_train = convert2npy(train_data)

nan_indices = np.argwhere(np.isnan(X_train))
print('#{} nan elements'.format(nan_indices))

inf_indices = np.argwhere(np.isinf(X_train))
print('#{} inf elements'.format(inf_indices))

#clf = tree.DecisionTreeClassifier()
clf = svm.LinearSVC()

clf.fit(X_train,Y_train)
joblib.dump(clf, 'IDS_classifier.joblib') 

X_val, y_val = convert2npy(val_data)

unique, counts = np.unique(y_val, return_counts=True)
print (np.asarray((unique, counts)).T)

y_pred = clf.predict(X_val)
print('val score: ',np.sum(y_pred==y_val)/len(y_val) )

X_test, y_test = convert2npy(test_data)
unique, counts = np.unique(y_test, return_counts=True)
print (np.asarray((unique, counts)).T)

y_pred = clf.predict(X_test)
print('test score: ',np.sum(y_pred==y_test)/len(y_test))
