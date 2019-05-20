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
    for file in glob.glob("../test/*.tsv"): test_files.append(file)
    for file in glob.glob("../training/tsv_data/*.tsv"): train_files.append(file)
    print('------\n', train_files, ' \n\n-----\n')
    run_algo()

def load_list_from_file():
    arr = []
    with open('speakers_all.data', 'rb') as filehandle:  
        arr = pickle.load(filehandle)
    return arr

def save_train_vec(fname, vec):
    fpath = '/Users/ibahadiraltun/Desktop/CLEF-2019-CheckThat/CLEF2019_Task1/data/submissions/solution_cont2/vectors/speakers/train/' + fname
    np.savetxt(fpath, vec, fmt = '%f')

def save_test_vec(fname, vec):
    fpath = '/Users/ibahadiraltun/Desktop/CLEF-2019-CheckThat/CLEF2019_Task1/data/submissions/solution_cont2/vectors/speakers/test/' + fname
    np.savetxt(fpath, vec, fmt = '%f')

def get_speaker_vector(speakers):
    speaker_list = load_list_from_file()
    sz = len(speaker_list)
    print(sz, '\n', speaker_list)
    vec_speaker = np.zeros((0, sz))
    for i in speakers:
        tmp = np.zeros((1, 0))
        for j in speaker_list:
            if i == j: tmp = np.concatenate((tmp, [[True]]), axis = 1)
            else: tmp = np.concatenate((tmp, [[False]]), axis = 1)
        vec_speaker = np.concatenate((vec_speaker, tmp), axis = 0)
    print(vec_speaker)
    return vec_speaker

def run_algo():
    for file in train_files:
        df = pd.read_csv(file, delimiter = '\t', encoding = 'utf-8', names = columns)
        vec = get_speaker_vector(df.speaker)
        save_train_vec(file.split('/')[3], vec)
    for file in test_files:
        df = pd.read_csv(file, delimiter = '\t', encoding = 'utf-8', names = ['linenumber', 'speaker', 'text'])
        vec = get_speaker_vector(df.speaker)
        save_test_vec(file.split('/')[2], vec)
    return None

if __name__ == '__main__':
    read_files()