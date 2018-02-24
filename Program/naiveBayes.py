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
        data=myfile.read().replace('\n', ' ')
    return data

'''
TrainMultinomialNB trains a machine learning model using the given training data specified as document tuple D, where D[0] is ham, and D[1] is spam, and classes are specified in class C
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
        s=0
        for token in v:
            t[c+'.'+token]=CountTokensOfTerm(text[c],token)
            s=s+CountTokensOfTerm(text[c],token)
        denominator = findSmoothedSum(v,c,t)
        for token in v:
            condProb[c+'.'+token] = (t[c+'.'+token]+1)/denominator
    return v, prior, condProb

'''
CountTokensOfTerm function counts the frequency of occurence of a given token in a given text
'''
def CountTokensOfTerm(text, token):
    text_words= text.split(" ")
    count=0
    for w in text_words:
        if w==token:
            count=count+1
    return count   
    
'''
findSmoothedSum finds the laplace smoothed sum used in denominator of condprob[t][c], given by ∑t′(Tct′+1)
'''    
def findSmoothedSum(v,c,t):
    sum=0
    for token in v:
       sum = sum+ t[c+'.'+token]  +1 
    return sum
  
'''
this function does what it's name suggests for given document tuple D and class c
'''      
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

'''
extractVocabulary function extracts the unique tokens given in D
'''       
def extractVocabulary(D):
    vocab=set()
    for entry in D[0]:
        word_list = entry.split(" ")
        vocab.update(set(word_list))
    for entry2 in D[1]:
        word_list2 = entry2.split(" ")
        vocab.update(set(word_list2))        
    return vocab
    
'''
countDocs counts number of documents in D
'''
def countDocs(D):
    return len(D[0])+len(D[1])
    
'''
countDocs counts number of documents in each class
'''
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
    res=TrainMultinomialNB(C,D)
#    print(res[0])
    print('=======================prior=========================')
    print(res[1])
    print('=======================conditional_probability=========================')
    print(res[2])
   