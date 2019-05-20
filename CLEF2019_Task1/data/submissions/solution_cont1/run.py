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
PATH = '/Users/ibahadiraltun/Desktop/CLEF-2019-CheckThat/CLEF2019_Task1/data/submissions/solution_primary/vectors/'
path_to_vectors = [PATH + 'categories/', PATH + 'named_entity/', PATH + 'part_of_speech/', PATH + 'bigram/']

columns = ['linenumber', 'speaker', 'text', 'label']
train_files = []
test_files = []

def read_files():
    for file in glob.glob("test/*.tsv"): test_files.append(file)
    for file in glob.glob("training/tsv_data/*.tsv"): train_files.append(file)
    print('------\n', train_files, ' \n\n-----\n')
    run_algo()

def run_algo():
    train_file = 'vectors/formatted_train.txt'
    os.system("java -jar RankLib.jar \
            -train {} -gmax 1 -ranker 0 -tree 50 -leaf 2 -metric2t MAP \
            -save model.txt".format(train_file))
    for file in test_files:
        fname = file.split('/')[1]
        test_file = 'vectors/formatted_X_test/' + fname
        score_file = 'results/unformatted_results/' + fname
        os.system("java -jar RankLib.jar -load model.txt -rank {} -metric2T MAP -score {}".format(test_file, score_file))
    return None

if __name__ == '__main__':
    read_files()