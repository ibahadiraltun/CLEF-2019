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

def get_qmark_vector(text):
    vec_qmark = np.zeros((0, 1))
    for i in text:
        qmark = 0
        if i.find('?') != -1: qmark = 1
        vec_qmark = np.concatenate((vec_qmark, [[qmark]]), axis = 0)
    print(len(vec_qmark), '\n', vec_qmark)
    return vec_qmark

def save_train_vec(fname, vec):
    fpath = '/Users/ibahadiraltun/Desktop/CLEF-2019-CheckThat/CLEF2019_Task1/data/submissions/solution_primary/vectors/qmark/train/' + fname
    np.savetxt(fpath, vec, fmt = '%f')

def save_test_vec(fname, vec):
    fpath = '/Users/ibahadiraltun/Desktop/CLEF-2019-CheckThat/CLEF2019_Task1/data/submissions/solution_primary/vectors/qmark/test/' + fname
    np.savetxt(fpath, vec, fmt = '%f')

def run_algo():
    for file in train_files:
        df = pd.read_csv(file, delimiter = '\t', encoding = 'utf-8', names = columns)
        vec = get_qmark_vector(df.text)
        save_train_vec(file.split('/')[3], vec)
    for file in test_files:
        df = pd.read_csv(file, delimiter = '\t', encoding = 'utf-8', names = ['linenumber', 'speaker', 'text'])
        vec = get_qmark_vector(df.text)
        save_test_vec(file.split('/')[2], vec)

    return None

if __name__ == '__main__':
    read_files()