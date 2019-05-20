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

def run_algo():
    y_train = np.loadtxt('vectors/y_train.txt', dtype = 'f')
    X_train = np.loadtxt('vectors/X_train.txt', dtype = 'f')
    print(len(y_train), ' ---- ', len(X_train[0]))
    with open('vectors/formatted_train.txt', 'w', encoding = 'utf-8') as out:
        line_number = 0
        for i in X_train:
            label = int(y_train[line_number])
            # if label == 1: print('!!!!!!!!!!!!!!!!!!!!!')
            out.write(str(label) + ' qid:1')
            cnt = 1
            for j in i:
                out.write(' ' + str(cnt) + ':' + str(j))
                cnt = cnt + 1
            out.write('\n')
            line_number = line_number + 1
    for file in test_files:
        fname = file.split('/')[1]
        X_test = np.loadtxt('vectors/X_test/' + fname, dtype = 'f')
        with open('vectors/formatted_X_test/' + fname, 'w', encoding = 'utf-8') as out:
            for i in X_test:
                out.write('0 qid:1')
                cnt = 1
                for j in i:
                    out.write(' ' + str(cnt) + ':' + str(j))
                    cnt = cnt + 1
                out.write('\n')

    return None

if __name__ == '__main__':
    read_files()