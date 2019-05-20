import collections
import pandas as pd
import numpy as np
import pickle
import glob
import sys
import os

PATH = '/Users/ibahadiraltun/Desktop/CLEF-2019-CheckThat/CLEF2019_Task1/data/submissions/solution_cont2/results/'
res_files = []

def read_files():
    for file in glob.glob("*.tsv"): res_files.append(file)
    print('------\n', res_files, ' \n\n-----\n')
    run_algo()

def run_algo():
    for file in res_files:
        fpath = PATH + file.split('/')[0]
        from format_checker.main import check_format
        if check_format(fpath): print('Format OK.')
        else: print('Format ERROR!')

if __name__ == '__main__':
    read_files()