#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 10:41:44 2019

@author: ibahadiraltun
"""

import pandas as pd
import glob


dataFiles = []


# adding columns to each tsvfile
for file in glob.glob("*.tsv"):
    fileLoc = "csv_data/" + file
    tsvFile = open(fileLoc, "w")
    tsvFile.write("linenumber\tspeaker\ttext\tlabel\n")
    with open(file) as tf:
        for line in tf.readlines():
            tsvFile.write(line)
    dataFiles.append(fileLoc)
    tsvFile.close()

#%%  Reading data as dataframe from tsvfile
for file in dataFiles:
    dataFrame = pd.read_csv(file, delimiter = '\t', encoding = 'utf-8')
#    print(dataFrame['text'])

    # from now on we will execute every dataframe

    # sorting with label values
    dfSortedByLabel = dataFrame.sort_values(by = ['label'], inplace = False, ascending = False)
#    print(dfSortedByLabel[['label', 'speaker', 'text']])

