# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 21:30:42 2018

@author: Mehak Beri
"""

import os

'''
getDocs function gets the documents inside the directories specified by hamPath and spamPath
'''

def getDocs(hamPath, spamPath):
    train_ham = os.listdir(hamPath)
    train_spam = os.listdir(spamPath)
    ham=[]
    spam=[]
    for file in train_ham:
        data=getText(hamPath+'/'+file)
        ham.append(data) 
    for files in train_spam:
        data1=getText(spamPath+'/'+files)
        spam.append(data1) 
    return ham,spam
    
'''
getText functions gets the text included in a file specified by the given filename
'''
def getText(filename):
    with open(filename, 'r', encoding="Latin-1") as myfile:
        data=myfile.read().replace('\n', '')
    return data

'''
TrainMultinomialNB trains a machine learning model using the given training data specified as documents D, and classes are specified in class C
'''
def TrainMultinomialNB(C,D):
    v = extractVocabulary(D)
    n = countDocs(D)
    prior={}
    text={}
    t={}
    condProb={}
    for c in C:
        Nc = countDocsInClass(D,c)
        prior[c]= Nc/n
        text[c]=ConcatenateTextOfAllDocsInClass(D,c)
        for token in v:
            t[c+'.'+token]=CountTokensOfTerm(text[c],token)
        denominator = findSmoothedSum(v,c)
        for token in v:
            condProb[c+'.'+token] = (t[c+'.'+token]+1)/denominator
    return v, prior, condProb

def CountTokensOfTerm(text, token):
    
    
def findSmoothedSum(v,c):
        
def ConcatenateTextOfAllDocsInClass(D,c):
    text=''
    if c=='ham':
        for docs in D[0]:
            text = text + docs + " "
        return text
    else:
        for docs in D[1]:
            text = text + docs + " "
        return text
        
def extractVocabulary(D):
    vocab=set()
    for entry in D[0]:
        word_list = entry.split(" ")
        vocab.update(set(word_list))
    for entry2 in D[1]:
        word_list2 = entry2.split(" ")
        vocab.update(set(word_list2))        
    return vocab
    
def countDocs(D):
    return len(D[0])+len(D[1])
    
def countDocsInClass(D,c):
    if c=='ham':
        return len(D[0])
    else:
        return len(D[1])

if __name__== '__main__':
    path_train_ham = '../dataSet1/train/ham'    
    path_train_spam= '../dataSet1/train/spam'
    D= getDocs(path_train_ham, path_train_spam)
    C=['ham','spam']
    TrainMultinomailNB(C,D)
   