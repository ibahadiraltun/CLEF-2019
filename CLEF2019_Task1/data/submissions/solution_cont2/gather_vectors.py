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
    for file in train_files:
        df = pd.read_csv(file, delimiter = '\t', encoding = 'utf-8', names = columns)
    for file in test_files:
        df = pd.read_csv(file, delimiter = '\t', encoding = 'utf-8', names = ['linenumber', 'speaker', 'text'])
    return None

if __name__ == '__main__':
    read_files()