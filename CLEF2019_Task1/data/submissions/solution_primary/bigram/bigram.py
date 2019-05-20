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

def get_bigram_list(text, labels):
    ind, mp = -1, collections.defaultdict(list)
    for line in text:
        ind, prev = ind + 1, ''
        for word in line.split(' '):
            mp[prev + word].append(labels[ind])
            prev = word
    
    global bigram_list
    label0, label1 = 0, 1
    for key, values in mp.items():
    #    print(key, ' ---> ', values)
        if (label0 in values and label1 not in values and len(values) >= 50) \
            or (label1 in values and label0 not in values and len(values) >= 50): bigram_list.append(key)
    
    # print(bigram_list)
    sz = len(bigram_list)
    print('-----> ', sz)
    
    with open('bigram_list.txt', 'w') as f:
        for i in bigram_list:
            f.write("%s\n" % i)

def get_bigram_vector(text):
    sz = len(bigram_list)
    vec_bigram = np.zeros((0, sz))
    for i in text:
        prev = ''
        mp = collections.defaultdict(int)
        for word in i.split(' '):
            mp[prev + word] = mp[prev + word] + 1
            prev = word
        vec_tmp = np.zeros((1, 0))
        for j in bigram_list:
            have = 0
            if mp[j] >= 1: have = 1
            vec_tmp = np.concatenate((vec_tmp, [[have]]), axis = 1)
        vec_bigram = np.concatenate((vec_bigram, vec_tmp), axis = 0)

    print(vec_bigram)
    return vec_bigram

def save_train_vec(fname, vec):
    fpath = '/Users/ibahadiraltun/Desktop/CLEF-2019-CheckThat/CLEF2019_Task1/data/submissions/solution_primary/vectors/bigram/train/' + fname
    np.savetxt(fpath, vec, fmt = '%f')

def save_test_vec(fname, vec):
    fpath = '/Users/ibahadiraltun/Desktop/CLEF-2019-CheckThat/CLEF2019_Task1/data/submissions/solution_primary/vectors/bigram/test/' + fname
    np.savetxt(fpath, vec, fmt = '%f')

def run_algo():
    df = pd.DataFrame()
    for file in train_files:
        tmp = pd.read_csv(file, delimiter = '\t', encoding = 'utf-8', names = columns)
        df = df.append(tmp, ignore_index = True)
    df.linenumber = df.index + 1
    print(len(df), '\n', df)
    get_bigram_list(df.text, df.label)
    for file in train_files:
        train_df = pd.read_csv(file, delimiter = '\t', encoding = 'utf-8', names = columns)
        vec = get_bigram_vector(train_df.text)
        save_train_vec(file.split('/')[3], vec)
    for file in test_files:
        test_df = pd.read_csv(file, delimiter = '\t', encoding = 'utf-8', names = ['linenumber', 'speaker', 'text'])
        vec = get_bigram_vector(test_df.text)
        save_test_vec(file.split('/')[2], vec)

    return None

if __name__ == '__main__':
    read_files()