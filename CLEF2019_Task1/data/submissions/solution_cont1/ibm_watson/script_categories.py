#%% template
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 19:32:11 2019

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
mp = collections.defaultdict()
vec_all = np.zeros((0, 0))
vec_cnt = 0

#%% input

def read_files():
    for file in glob.glob("categories_test/*.tsv"):
        fileLoc = file
        test_files.append(fileLoc)
    for file in glob.glob("../training/categories_train/*.tsv"):
        fileLoc = file
        train_files.append(fileLoc)
    print('------\n', train_files, ' \n\n-----\n')
    run_algo()

#%% algorithm

def remove_last_column(x):
    return x[..., :-1]

def save_vec_to_file(vec, fname):
    np.savetxt('/Users/ibahadiraltun/Desktop/CLEF-2019-CheckThat/CLEF2019_Task1/data/submissions/solution_primary/vectors/categories/test/{}'.format(fname), 
        vec, fmt = '%f')

def save_vec_to_file2(vec, fname):
    np.savetxt('/Users/ibahadiraltun/Desktop/CLEF-2019-CheckThat/CLEF2019_Task1/data/submissions/solution_primary/vectors/categories/train/{}'.format(fname), 
        vec, fmt = '%f')

def update(val):
    if val == 'NO_LABEL': mp[val] = True
    else:
        tmp = val.split('/')
        if len(tmp) > 2: label = tmp[1] + tmp[2]
        else: label = tmp[1]
        mp[label] = True
    # mp[val] = True

def analyse(fpath):
    # fpath = '/Users/ibahadiraltun/Desktop/CLEF-2019-CheckThat/CLEF2019_Task1/data/submissions/solution_primary/ibm_watson/categories/' + fname
    print(fpath)
    with open(fpath, 'r') as file:
        for line in file:
            if line == 'NO_LABEL\n': update('NO_LABEL')
            else:
                line = line.replace(' ', '')
                all_categ = line.split('|')
                for categ in all_categ:
                    if categ == '\n': continue
                    label = categ.split(';')[1]
                    update(label)
    print('======= NEW FILE =======')

def score(fpath):
    global vec_all
    with open(fpath, 'r') as file:
        for line in file:
            line = line.replace(' ', '')
            line = line[:-1]
            mx_score = collections.defaultdict(float)
            if line == 'NO_LABEL': mx_score[line] = 1.0
            else:
                for categ in line.split('|'):
                    if categ == '': continue
                    score = categ.split(';')[0]
                    score = float(score)
                    tmp = categ.split(';')[1].split('/')
                    if len(tmp) > 2: label = tmp[1] + tmp[2]
                    else: label = tmp[1]
                    # label = categ.split(';')[1]
                    mx_score[label] = max(mx_score[label], score)
            vec_tmp = np.zeros((1, 0))
            for categs in mp.keys():
                eq = 0
                for ctemp, _ in mx_score.items():
                    if categs == ctemp: eq = _
                vec_tmp = np.concatenate((vec_tmp, [[eq]]), axis = 1)
            vec_all = np.concatenate((vec_all, vec_tmp), axis = 0)

def run_algo():
    for file in test_files:
        fpath = '/Users/ibahadiraltun/Desktop/CLEF-2019-CheckThat/CLEF2019_Task1/data/submissions/solution_primary/ibm_watson/categories_test/' + file.split('/')[1]
        print(fpath)
        analyse(fpath)
    for file in train_files:
        fpath = '/Users/ibahadiraltun/Desktop/CLEF-2019-CheckThat/CLEF2019_Task1/data/submissions/solution_primary/training/categories_train/' + file.split('/')[3]
        print(fpath)
        analyse(fpath)

    for key in mp.keys(): print(key)
    global vec_all, vec_cnt
    sz = len(mp.keys())
    vec_all = np.zeros((0, sz))

    for file in test_files:
        fpath = '/Users/ibahadiraltun/Desktop/CLEF-2019-CheckThat/CLEF2019_Task1/data/submissions/solution_primary/ibm_watson/categories_test/' + file.split('/')[1]
        # for test file
        score(fpath)
        print(len(vec_all), '\n', vec_all)
        save_vec_to_file(vec_all, file.split('/')[1])
        vec_all = np.zeros((0, sz))

    vec_all = np.zeros((0, sz))
    for file in train_files:
        fpath = '/Users/ibahadiraltun/Desktop/CLEF-2019-CheckThat/CLEF2019_Task1/data/submissions/solution_primary/training/categories_train/' + file.split('/')[3]
        # for train file
        score(fpath)
        print(len(vec_all), '\n', vec_all)
        save_vec_to_file2(vec_all, file.split('/')[3])
        vec_all = np.zeros((0, sz))

    return None

#%% main

if __name__ == '__main__':
    read_files()
