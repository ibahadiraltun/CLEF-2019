#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 18:02:32 2019

@author: ibahadiraltun
"""
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import glob

dfiles = []

for file in glob.glob("../training/csv_data/*.tsv"):
    fileLoc = file
    dfiles.append(fileLoc)

# columns = ['linenumber', 'speaker', 'text', 'label']

#%% input section
#for file in dfiles:
file = dfiles[0]
df = pd.read_csv(file, delimiter = '\t', encoding = 'utf-8')

df.label = df.label.astype(float)

features = ['word_number', 'feature2', 'feature3', 'feature4']

# initializing X matrix with Nx1 dimensions (only word_number column)

X = np.zeros((0, 1))

for i in df.text:
    cnt_word = i.count(' ') + 1
    tmp = [[cnt_word]]
    X = np.concatenate((X, tmp), axis = 0)

# initializing y matrix with same dimensions as X
y = np.array(df.label)
#%% logistic regression

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0,random_state=0)

X_test = X

logreg = LogisticRegression()

logreg.train(X_train,y_train)

y_pred=logreg.predict(X_test)

probs = logreg.predict_proba(X_test)

#%% output section

outFile = "results/" + file.split('/')[3].split('.')[0] + "_output.tsv"
out = open(outFile, "w")

line_number = 0
for i in y_pred:
    line_number = line_number + 1
    class_prob = probs[line_number - 1][0]
    if i == 0:
        class_prob = 1 - class_prob
    out.write(str(line_number) + "\t" + str(class_prob) + "\n")

