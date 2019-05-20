import collections
import pandas as pd
import numpy as np
import pickle
import glob
import sys
import os

import json
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_watson.natural_language_understanding_v1 import Features, CategoriesOptions

API_KEY = 0 #
URL = 'https://gateway-lon.watsonplatform.net/natural-language-understanding/api'

natural_language_understanding = NaturalLanguageUnderstandingV1(
    version = '2018-11-16',
    iam_apikey = API_KEY,
    url = URL
)

dfiles = []
vec_cnt = 0

def read_files():
    for file in glob.glob("../test/*.tsv"):
        fileLoc = file
        dfiles.append(fileLoc)
    run_script()

def get_response(text):
    response = natural_language_understanding.analyze(
        text = text,
        features = Features(categories = CategoriesOptions(limit = 3)),
        language = 'en'
    ).get_result()

    # print(json.dumps(response, indent = 2))
    return response

def write_to_file(file, res):
    fpath = '/Users/ibahadiraltun/Desktop/CLEF-2019-CheckThat/CLEF2019_Task1/data/submissions/solution_primary/ibm_watson/categories/' + file.split('/')[2]
    with open(fpath, "w", encoding = 'utf-8') as f: f.write(res)

def run_script():
    all_cnt = 0
    mp = collections.defaultdict(list)
    for file in dfiles:
        all_cnt = all_cnt + 1
        print("\n\n----reading {} as test file----\n".format(file))
        df = pd.read_csv(file, delimiter = '\t', encoding = 'utf-8', names = ['linenumber', 'speaker', 'text', 'label'])
        text = df.text
        all_lines = ''
        cnt = 1
        sz = len(text)
        for line in text:
            print(cnt, sz)
            cnt = cnt + 1
            if line in mp.keys():
                print(line, ' ----> ', mp[line])
                cur = mp[line]
            else:
                print(line)
                res = get_response(line)
                categs = res['categories']
                cur = ''
                for i in categs:
                    cur = cur + str(i['score']) + '; ' + i['label'] + '|'
                if cur == '': cur = 'NO_LABEL'
                print(cur)
                mp[line] = cur
            all_lines = all_lines + cur + '\n'
            # print(cur)
        #    write_to_file(file, all_lines)
        write_to_file(file, all_lines)
        print(' ----- ', file, ' ----- ENDED')
        # sys.exit()

if __name__ == '__main__':
    read_files()
