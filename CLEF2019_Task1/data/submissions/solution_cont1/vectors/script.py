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
    for file in glob.glob("../test/*.tsv"): test_files.append(file)
    for file in glob.glob("../training/tsv_data/*.tsv"): train_files.append(file)
    print('------\n', train_files, ' \n\n-----\n')
    run_algo()

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
    for file in test_files:
        fname = file.split('/')[2]
        vec_prev = np.loadtxt('X_test/' + fname, dtype = 'f')
        vec_speaker = np.loadtxt('speakers/test/' + fname, dtype = 'f')
        vec_cur = np.concatenate((vec_prev, vec_speaker), axis = 1)
        save_X_test_vector(fname, vec_cur)
    vec_all = np.zeros((0, 33))
    for file in train_files:
        fname = file.split('/')[3]
        vec = np.loadtxt('speakers/train/' + fname, dtype = 'f')
        vec_all = np.concatenate((vec_all, vec), axis = 0)
    vec_prev = np.loadtxt('X_train.txt', dtype = 'f')
    vec_cur = np.concatenate((vec_prev, vec_all), axis = 1)
    save_X_train_vector(vec_cur)
    return None

if __name__ == '__main__':
    read_files()