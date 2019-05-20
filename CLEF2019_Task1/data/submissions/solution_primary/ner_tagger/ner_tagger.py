#%% ner_tagger
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 4 20:56:11 2019

@author: ibahadiraltun
"""


from nltk.internals import config_java
from nltk.tokenize import word_tokenize
from nltk.tag.stanford import StanfordNERTagger
from nltk.tag.stanford import StanfordPOSTagger
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from tempfile import TemporaryFile

import collections
import pandas as pd
import numpy as np
import pickle
import glob
import sys
import os

jar = './stanford-ner-tagger/ner-tagger.jar'
model = './stanford-ner-tagger/english.muc.7class.distsim.crf.ser.gz'
ntagger = StanfordNERTagger(model, jar, encoding = 'utf8')

test_files = []
train_files = []
columns = ['linenumber', 'speaker', 'text', 'label']

def read_files():
    for file in glob.glob("../test/*.tsv"):
        fileLoc = file
        test_files.append(fileLoc)
    for file in glob.glob("../training/tsv_data/*.tsv"):
        fileLoc = file
        train_files.append(fileLoc)
    print('------\n', train_files, ' \n\n-----\n')
    run_algo()

def tag_file(file):
    df = pd.read_csv(file, delimiter = '\t', encoding = 'utf-8', names = columns)
    text = df.text
    cnt, all_text = 1, ''
    for i in text:
        all_text = all_text + ' ' + i + ' sp.{} '.format(cnt)
        cnt = cnt + 1

    words = word_tokenize(all_text)
    tagged_words = ntagger.tag(words)

    cur_sp = 1
    cnt_loc = cnt_per = cnt_org = cnt_mon = cnt_perc = cnt_date = cnt_time = 0
    vec_loc = vec_per = vec_org = vec_mon = vec_perc = vec_date = vec_time = np.zeros((0, 1))
    for i in tagged_words:
        if i[0] == 'sp.{}'.format(cur_sp):
            vec_loc = np.concatenate((vec_loc, [[cnt_loc]]), axis = 0)
            vec_per = np.concatenate((vec_per, [[cnt_per]]), axis = 0)
            vec_org = np.concatenate((vec_org, [[cnt_org]]), axis = 0)
            vec_mon = np.concatenate((vec_mon, [[cnt_mon]]), axis = 0)
            vec_perc = np.concatenate((vec_perc, [[cnt_perc]]), axis = 0)
            vec_date = np.concatenate((vec_date, [[cnt_date]]), axis = 0)
            vec_time = np.concatenate((vec_time, [[cnt_time]]), axis = 0)
            cur_sp = cur_sp + 1
            cnt_loc = cnt_per = cnt_org = cnt_mon = cnt_perc = cnt_date = cnt_time = 0
        elif i[1] == 'PERSON': cnt_per = cnt_per + 1
        elif i[1] == 'LOCATION': cnt_loc = cnt_loc + 1
        elif i[1] == 'ORGANIZATION': cnt_org = cnt_org + 1
        elif i[1] == 'MONEY': cnt_mon = cnt_mon + 1
        elif i[1] == 'PERCENT': cnt_perc = cnt_perc + 1
        elif i[1] == 'DATE': cnt_date = cnt_date + 1
        elif i[1] == 'TIME': cnt_time = cnt_time + 1

    vec_all = np.concatenate((vec_loc, vec_per, vec_org, vec_mon, vec_perc, vec_date, vec_time), axis = 1)
    print(file, '\n', vec_all)
    return vec_all

def save_test_vector(vec, fname):
    fpath = '/Users/ibahadiraltun/Desktop/CLEF-2019-CheckThat/CLEF2019_Task1/data/submissions/solution_primary/vectors/named_entity/test/' + fname
    np.savetxt(fpath, vec, fmt = '%f')

def save_train_vector(vec, fname):
    fpath = '/Users/ibahadiraltun/Desktop/CLEF-2019-CheckThat/CLEF2019_Task1/data/submissions/solution_primary/vectors/named_entity/train/' + fname
    np.savetxt(fpath, vec, fmt = '%f')

def run_algo():
    for file in test_files:
        vec = tag_file(file)
        save_test_vector(vec, file.split('/')[2])
    for file in train_files:
        vec = tag_file(file)
        save_train_vector(vec, file.split('/')[3])
    return None

if __name__ == '__main__':
    read_files()