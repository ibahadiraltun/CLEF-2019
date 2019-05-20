#%% template
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 18 11:32:11 2019

@author: ibahadiraltun
"""

import collections
import pandas as pd
import numpy as np
import pickle
import glob
import sys
import os

DIR_PATH = os.getcwd()
PATH = '/Users/ibahadiraltun/Desktop/CLEF-2019-CheckThat/CLEF2019_Task1/data/submissions/solution_cont1/vectors/'
path_to_vectors = [PATH + 'categories/', PATH + 'named_entity/', PATH + 'part_of_speech/', PATH + 'bigram/']

columns = ['linenumber', 'speaker', 'text', 'label']
train_files = []
test_files = []
bigram_list = []
mp = collections.defaultdict()
vec_all = np.zeros((0, 0))
vec_cnt = 0

def read_files():
    for file in glob.glob("test/*.tsv"): test_files.append(file)
    for file in glob.glob("training/tsv_data/*.tsv"): train_files.append(file)
    print('------\n', train_files, ' \n\n-----\n')
    run_algo()

def get_y_train_vector():
    df = pd.DataFrame()
    for file in train_files:
        tmp = pd.read_csv(file, delimiter = '\t', encoding = 'utf-8', names = columns)
        df = df.append(tmp, ignore_index = True)
    labels = df.label
    vec = np.zeros((0, 1))
    for i in labels:
        vec = np.concatenate((vec, [[i]]), axis = 0)
    return vec

def get_X_train_vector():
    vec_all = np.zeros((0, 388))
    for file in train_files:
        fname = file.split('/')[2]
        tmp_df = pd.read_csv(file, delimiter = '\t', encoding = 'utf-8', names = ['linenumber', 'speaker', 'text'])
        vec_cur = np.zeros((len(tmp_df), 0))
        for vecs in path_to_vectors:
            path = vecs + 'train/' + fname
            vec_tmp = np.loadtxt(path, dtype = 'f')
            vec_cur = np.concatenate((vec_cur, vec_tmp), axis = 1)
            print(len(vec_cur), '---', len(vec_tmp))
        vec_all = np.concatenate((vec_all, vec_cur), axis = 0)
    print('----- X_train vector ----- ', len(vec_all), '\n', vec_all)
    return vec_all

def get_X_test_vector(fname):
    tmp_df = pd.read_csv('test/' + fname, delimiter = '\t', encoding = 'utf-8', names = ['linenumber', 'speaker', 'text'])
    vec_all = np.zeros((len(tmp_df), 0))
    for vecs in path_to_vectors:
        path = vecs + 'test/' + fname
        vec_tmp = np.loadtxt(path, dtype = 'f')
        vec_all = np.concatenate((vec_all, vec_tmp), axis = 1)
    print('----- X_test vector ----- ', len(vec_all), '\n', vec_all)
    return vec_all

def save_y_train_vector(vec):
    fpath = '/Users/ibahadiraltun/Desktop/CLEF-2019-CheckThat/CLEF2019_Task1/data/submissions/solution_cont1/vectors/y_train.txt'
    np.savetxt(fpath, vec, fmt = '%f')

def save_X_train_vector(vec):
    fpath = '/Users/ibahadiraltun/Desktop/CLEF-2019-CheckThat/CLEF2019_Task1/data/submissions/solution_cont1/vectors/X_train.txt'
    np.savetxt(fpath, vec, fmt = '%f')

def save_X_test_vector(fname, vec):
    fpath = '/Users/ibahadiraltun/Desktop/CLEF-2019-CheckThat/CLEF2019_Task1/data/submissions/solution_cont1/vectors/X_test/' + fname
    np.savetxt(fpath, vec, fmt = '%f')

def run_algo():
    y_train = get_y_train_vector()
    X_train = get_X_train_vector()
    save_y_train_vector(y_train)
    save_X_train_vector(X_train)
    for file in test_files:
        vec = get_X_test_vector(file.split('/')[1])
        save_X_test_vector(file.split('/')[1], vec)
    return None

if __name__ == '__main__':
    read_files()