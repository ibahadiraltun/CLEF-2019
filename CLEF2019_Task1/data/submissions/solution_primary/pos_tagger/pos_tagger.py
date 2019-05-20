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

pos_jar = './stanford-pos-tagger/stanford-postagger.jar'
pos_model = './stanford-pos-tagger/english-left3words-distsim.tagger'
ptagger = StanfordPOSTagger(pos_model, pos_jar)

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

def empty_pos_dict():
    tmp = {
        'CC': 0, 'CD': 0, 'DT': 0, 'EX': 0, 'FW': 0, 'IN': 0,'JJ': 0,
        'JJR': 0, 'JJS': 0, 'LS': 0, 'MD': 0, 'NN': 0, 'NNS': 0, 'NNP': 0,
        'NNPS': 0, 'PDT': 0, 'POS': 0, 'PRP': 0, 'PRP$': 0, 'RB': 0, 'RBR': 0,
        'RBS': 0, 'RP': 0, 'SYM': 0, 'TO': 0, 'UH': 0, 'VB': 0, 'VBD': 0,
        'VBG': 0, 'VBN': 0, 'VBP': 0, 'VBZ': 0, 'WDT': 0, 'WP': 0, 'WP$': 0, 'WRB': 0
    }
    return tmp

def tag_file(file):
    df = pd.read_csv(file, delimiter = '\t', encoding = 'utf-8', names = columns)
    text = df.text
    cnt, all_text = 1, ''
    for i in text:
        all_text = all_text + ' ' + i + ' sp.{} '.format(cnt)
        cnt = cnt + 1
    
    cur_sp = 1
    vec_pos = np.zeros((0, 36))
    words = word_tokenize(all_text)
    tagged_words = ptagger.tag(words)
    pos_all = empty_pos_dict()
    for i in tagged_words:
        if i[0] == 'sp.{}'.format(cur_sp):
            vec_tmp = np.zeros((1, 0))
            for _, value in pos_all.items():
                vec_tmp = np.concatenate((vec_tmp, [[value]]), axis = 1)
            vec_pos = np.concatenate((vec_pos, vec_tmp), axis = 0)
            pos_all = empty_pos_dict()
            cur_sp = cur_sp + 1
        elif i[1] in pos_all.keys(): pos_all[i[1]] = pos_all[i[1]] + 1

    print(file, '\n', len(vec_pos), '\n',vec_pos)
    return vec_pos

def save_test_vector(vec, fname):
    fpath = '/Users/ibahadiraltun/Desktop/CLEF-2019-CheckThat/CLEF2019_Task1/data/submissions/solution_primary/vectors/part_of_speech/test/' + fname
    np.savetxt(fpath, vec, fmt = '%f')

def save_train_vector(vec, fname):
    fpath = '/Users/ibahadiraltun/Desktop/CLEF-2019-CheckThat/CLEF2019_Task1/data/submissions/solution_primary/vectors/part_of_speech/train/' + fname
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