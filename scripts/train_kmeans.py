#!/bin/python 

import numpy as np
import os
import pandas as pd
import glob
from sklearn.cluster.k_means_ import KMeans
from scipy.cluster.vq import kmeans, whiten
import cPickle
import pickle
import sys

# Performs K-means clustering and save the model to a local file

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print "Usage: {0} mfcc_csv_file cluster_num output_file".format(sys.argv[0])
        print "mfcc_csv_file -- path to the mfcc csv file"
        print "cluster_num -- number of cluster"
        print "output_file -- path to save the k-means model"
        exit(1)

    mfcc_csv_file = sys.argv[1]; output_file = sys.argv[3]
    cluster_num = int(sys.argv[2])
    
    select_mfcc_csv = pd.read_csv(mfcc_csv_file, delimiter=';', header=None)
    select_mfcc_csv_data = select_mfcc_csv.values
    select_mfcc_csv_data = whiten(select_mfcc_csv)
    
    pkl_filename = "KMeans_50centers.pkl"
    
    # Load from file
    with open(pkl_filename, 'rb') as file:
        pickle_model = pickle.load(file)
        
    output_file = pickle_model
    
    print "K-means trained successfully!"
