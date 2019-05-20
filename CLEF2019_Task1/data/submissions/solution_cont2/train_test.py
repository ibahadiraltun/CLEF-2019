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
PATH = '/Users/ibahadiraltun/Desktop/CLEF-2019-CheckThat/CLEF2019_Task1/data/submissions/solution_cont2/vectors/'
path_to_vectors = [PATH + 'categories/', PATH + 'named_entity/', PATH + 'part_of_speech/', PATH + 'bigram/', PATH + 'speakers/']

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
    vec_speaker = np.zeros((0, 33))
    for file in train_files:
        fname = file.split('/')[2]
        vec = np.loadtxt('vectors/speakers/train/' + fname, dtype = 'f')
        print(len(vec), ' ', len(X_train))
        vec_speaker = np.concatenate((vec_speaker, vec), axis = 0)
    X_train = np.concatenate((X_train, vec_speaker), axis = 1)
    from sklearn.linear_model import LogisticRegression
    logreg = LogisticRegression(max_iter = 250)
    logreg.fit(X_train, y_train)
    for file in test_files:
        fname = file.split('/')[1]
        df = pd.read_csv(file, delimiter = '\t', encoding = 'utf-8', names = ['linenumber', 'speaker', 'text'])
        X_test = np.loadtxt('vectors/X_test/' + fname, dtype = 'f')
        vec_speaker = np.loadtxt('vectors/speakers/test/' + fname, dtype = 'f')
        X_test = np.concatenate((X_test, vec_speaker), axis = 1)
        print(X_test)
        preds = logreg.predict(X_test)
        probs = logreg.predict_proba(X_test)
        print_results(fname, preds, probs, df.text, df.speaker)
    return None

def print_results(fname, preds, probs, text, speakers):
    with open('results/' + fname, "w", encoding = 'utf-8') as out:
        line_number = 0
        for i in preds:
            if text[line_number].lower() == 'thank you' \
                or text[line_number].lower() == 'welcome' \
                or text[line_number].lower() == 'goodbye' \
                or speakers[line_number].lower() == 'system' \
                or speakers[line_number].lower() == 'system (applause)':
                    probs[line_number][1] = 0.0
                    print(line_number, ' ------BINGOOOOO------')
            line_number = line_number + 1
            class_prob = probs[line_number - 1][1]
        #   SVM ---> class_prob = probs[line_number - 1]
            if i == 1: print("\n------- {}   ----   {}".format(i, class_prob))
        #    if i == 0: class_prob = 1 - class_prob
            line_number_str = "{}".format(line_number)
            class_prob_str = "{:.6f}".format(class_prob)
            out.write(line_number_str + "\t" + class_prob_str + "\n")

if __name__ == '__main__':
    read_files()