#%% template
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 11:55:23 2019

@author: ibahadiraltun
"""

from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import glob
import os


DIR_PATH = os.getcwd()

dfiles = []

columns = ['linenumber', 'speaker', 'text', 'label']


def get_vector(text):
    vec = np.zeros((0, 1))
    for i in text:
        if type(i).__name__ == 'float': val = i
        else: val = i.count(' ') + 1
        tmp = [[val]]
        vec = np.concatenate((vec, tmp), axis = 0)
    return vec

#%% input

def read_files():
    for file in glob.glob("../training/csv_data/*.tsv"):
        fileLoc = file
        dfiles.append(fileLoc)
    run_algo()

#%% algorithm

def run_algo():
    all_avg_LR = all_avg_random = all_avg_ngram = 0
    for file in dfiles: 
        print("\n\n----reading {} as test file----\n".format(file))
        test_df = pd.read_csv(file, delimiter = '\t', encoding = 'utf-8')
        test_df.label = test_df.label.astype(float)
        X_test_vector = get_vector(test_df.text)
        y_test_vector = get_vector(test_df.label)
        
        train_df = pd.DataFrame(columns = columns)
        for train_file in dfiles:
            if train_file != file:
                tmp = pd.read_csv(train_file, delimiter = '\t', encoding = 'utf-8')
                train_df = train_df.append(tmp, ignore_index = True)
            
        train_df.linenumber = train_df.index + 1
        train_df.label = train_df.label.astype(float)
        X_train_vector = get_vector(train_df.text)
        y_train_vector = get_vector(train_df.label)
        
#        print(len(X_train_vector), len(y_train_vector))
#        
#        print("gold_path ----> ", file)
        
        tmp_dir_path = DIR_PATH[: DIR_PATH.index("/solution")]
        
        preds, probs = run_LR(X_train_vector, X_test_vector, y_train_vector, y_test_vector)
        gold_fpath = tmp_dir_path + "/training/tsv_data/" + file.split('/')[3]
        pred_fpath = tmp_dir_path + "/solution/results/" + file.split('/')[3].split('.')[0] + "_pred.tsv"
                
#        print(tmp_dir_path)
        
        avg_LR, avg_random, avg_ngram = print_results(preds, probs, gold_fpath, pred_fpath)
        all_avg_LR = all_avg_LR + avg_LR
        all_avg_random = all_avg_random + avg_random
        all_avg_ngram = all_avg_ngram + avg_ngram
    
    all_avg_LR = all_avg_LR / len(dfiles)
    all_avg_random = all_avg_random / len(dfiles)
    all_avg_ngram = all_avg_ngram / len(dfiles)
    
    print("\n----- ALL AVERAGE SCORES -----\n")
    print("Logistic Regression AVGP ----> ", all_avg_LR)
    print("Random baseline AVGP ----> ", all_avg_random)
    print("Ngram baseline AVGP ----> ", all_avg_ngram)


#%% logistic regression

def run_LR(X_train, X_test, y_train, y_test):
    
    y_test = y_test.ravel()
    y_train = y_train.ravel()
    
    print("\n\nlength y_train && X_train ---> ", len(y_train), len(X_train))
    print("length y_test && X_test ---> ", len(y_test), len(X_test))
    
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    preds = logreg.predict(X_test)
    probs = logreg.predict_proba(X_test)
    
    print(len(preds), len(probs))
    
    return preds, probs

#%% output

def print_results(preds, probs, gold_fpath, pred_fpath):
    
    with open(pred_fpath, "w", encoding = 'utf-8') as out:
        line_number = 0
        for i in preds:
            line_number = line_number + 1
            class_prob = probs[line_number - 1][1]
            if i == 1: print("\n------- {}   ----   {}".format(i, class_prob))
        #    if i == 0: class_prob = 1 - class_prob
            line_number_str = "{}".format(line_number)
            class_prob_str = "{:.4f}".format(class_prob)
            out.write(line_number_str + "\t" + class_prob_str + "\n")
    
    print("\n\n------FILES------ \n\n gold_file ---> {} \n pred_file ---> {}".format(gold_fpath, pred_fpath))
    
    return get_score(gold_fpath, pred_fpath)

#%% scorer

def get_score(gold_fpath, pred_fpath):
    
#    print('\n\n', DIR_PATH)
    print("\n\n----- SCORING -----")
    
    from format_checker.main import check_format
    from scorer.main import evaluate
    from baselines.baselines import run_baselines
        
    if check_format(pred_fpath):
        _, _, avg_LR, _, _ = evaluate(gold_fpath, pred_fpath)
        print("\nLogistic Regression AVGP ----> ", avg_LR)
        avg_random, avg_ngram = run_baselines(gold_fpath)
    else:
        print("\ncheck_format ERROR!!! -> (sol_template2.get_score)")

    return avg_LR, avg_random, avg_ngram

#%% main

if __name__ == '__main__':
    read_files()

