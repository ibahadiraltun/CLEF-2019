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
bigram_list = []
mp = collections.defaultdict()
vec_all = np.zeros((0, 0))
vec_cnt = 0

def read_files():
    for file in glob.glob("../test/*.tsv"): test_files.append(file)
    for file in glob.glob("../training/tsv_data/*.tsv"): train_files.append(file)
    print('------\n', train_files, ' \n\n-----\n')
    run_algo()

def run_algo():
    for file in test_files:
        fname = file.split('/')[2]
        df = pd.read_csv(file, delimiter = '\t', encoding = 'utf-8', names = ['linenumber', 'speaker', 'text'])
        text, speakers = df.text, df.speaker
        result = ''
        with open('unformatted_results/' + fname, 'r', encoding = 'utf-8') as f:
            for lines in f.readlines():
                line_number = int(lines.split('\t')[1]) + 1
                score = str(lines.split('\t')[2])
                score = score[:9]
                if text[line_number - 1].lower() == 'thank you' \
                    or text[line_number - 1].lower() == 'welcome' \
                    or text[line_number - 1].lower() == 'goodbye' \
                    or speakers[line_number - 1].lower() == 'system' \
                    or speakers[line_number - 1].lower() == 'system (applause)':
                        print('---- BINGOOO ----')
                        score = "0.0"
                result = result + str(line_number) + '\t' + score + '\n'
        with open('formatted_results/' + fname, 'w', encoding = 'utf-8') as out:
            out.write(result)
    return None

if __name__ == '__main__':
    read_files()