#!/bin/python
import numpy
import os
import cPickle
from sklearn.cluster.k_means_ import KMeans
import sys
# Generate k-means features for videos; each video is represented by a single vector

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print "Usage: {0} kmeans_model, cluster_num, file_list".format(sys.argv[0])
        print "kmeans_model -- path to the kmeans model"
        print "cluster_num -- number of cluster"
        print "file_list -- the list of videos"
        exit(1)

    kmeans_model = sys.argv[1]; file_list = sys.argv[3]
    cluster_num = int(sys.argv[2])

    # load the kmeans model
    kmeans = cPickle.load(open(kmeans_model,"rb"))
    
    train_audio_list = pd.read_csv('./list/train.video', header=None)
    val_audio_list = pd.read_csv('./list/val.video', header=None)
    test_audio_list = pd.read_csv('./list/test.video', header=None)

    train_audio_files_idx = []

    for i in train_audio_list.values:
        temp = []
        for j in range(len(files)):
            temp.append(i[0] in files[j].replace('./mfcc/', '').replace('.mfcc.csv', ''))

        if np.array(temp).any() == True:
            train_audio_files_idx.append(np.where(np.array(temp) == True)[0][0])
        else:
            pass

    val_audio_files_idx = []

    for i in val_audio_list.values:
        temp = []
        for j in range(len(files)):
            temp.append(i[0] in files[j].replace('./mfcc/', '').replace('.mfcc.csv', ''))

        if np.array(temp).any() == True:
            val_audio_files_idx.append(np.where(np.array(temp) == True)[0][0])
        else:
            pass

    test_audio_files_idx = []

    for i in test_audio_list.values:
        temp = []
        for j in range(len(files)):
            temp.append(i[0] in files[j].replace('./mfcc/', '').replace('.mfcc.csv', ''))

        if np.array(temp).any() == True:
            test_audio_files_idx.append(np.where(np.array(temp) == True)[0][0])
        else:
            pass

    train_audio_files_idx = np.array(train_audio_files_idx)
    train_audio_files = np.array(files)[train_audio_files_idx]

    val_audio_files_idx = np.array(val_audio_files_idx)
    val_audio_files = np.array(files)[val_audio_files_idx]

    test_audio_files_idx = np.array(test_audio_files_idx)
    test_audio_files = np.array(files)[test_audio_files_idx]

    train_data_labels = pd.read_csv('./all_trn.lst', delimiter=' ', header=None, names = ['file_id', 'class'])
    train_data_labels.replace('P001', 1, inplace=True)
    train_data_labels.replace('P002', 2, inplace=True)
    train_data_labels.replace('P003', 3, inplace=True)
    train_data_labels.fillna(0, inplace=True)

    val_data_labels = pd.read_csv('./all_val.lst', delimiter=' ', header=None, names = ['file_id', 'class'])
    val_data_labels.replace('P001', 1, inplace=True)
    val_data_labels.replace('P002', 2, inplace=True)
    val_data_labels.replace('P003', 3, inplace=True)
    val_data_labels.fillna(0, inplace=True)

    # Match labels
    train_labels_index = []

    for i in range(len(train_audio_files)):
        try:
            temp = train_data_labels.loc[train_data_labels['file_id'] == train_audio_files[i].replace('./mfcc/', '').replace('.mfcc.csv', '')]
            train_labels_index.append(temp.index[0])
        except:
            pass

    train_labels_index = np.array(train_labels_index)

    val_labels_index = []

    for i in range(len(val_audio_files)):
        try:
            temp = val_data_labels.loc[val_data_labels['file_id'] == val_audio_files[i].replace('./mfcc/', '').replace('.mfcc.csv', '')]
            val_labels_index.append(temp.index[0])
        except:
            pass

    val_labels_index = np.array(val_labels_index)

    train_labels = train_data_labels.values[:,1][train_labels_index].astype(int)
    train_id = train_data_labels.values[:,0][train_labels_index]

    val_labels = val_data_labels.values[:,1][val_labels_index].astype(int)
    val_id = val_data_labels.values[:,0][val_labels_index]

    train_input_svm = []

    for i in train_id:
        temp = np.load('./train_data/' + 'train_data' + str(i[3:]) + '.npy')
        train_pred_centers = pickle_model.predict(temp)
        occurence_count = Counter(train_pred_centers)
        train_input_svm.append(occurence_count)

    train_input_svm_ = []

    for i in range(len(train_input_svm)):
        temp = np.zeros(50)
        temp[np.array(train_input_svm[i].keys())] = np.array(train_input_svm[i].values())
        temp_ = temp.reshape(1, -1)
        train_input_svm_.append(temp_[0])

    val_input_svm = []

    for i in val_id:
        temp = np.load('./val_data/' + 'val_data' + str(i[3:]) + '.npy')
        val_pred_centers = pickle_model.predict(temp)
        occurence_count = Counter(val_pred_centers)
        val_input_svm.append(occurence_count)

    val_input_svm_ = []

    for i in range(len(val_input_svm)):
        temp = np.zeros(50)
        temp[np.array(val_input_svm[i].keys())] = np.array(val_input_svm[i].values())
        temp_ = temp.reshape(1, -1)
        val_input_svm_.append(temp_[0])
    
    print "K-means features generated successfully!"
