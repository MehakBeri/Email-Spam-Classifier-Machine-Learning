# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 21:30:42 2018

@author: Mehak Beri
"""

import os
import math
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
    s_sum={}
    for c in C:
        Nc = countDocsInClass(D,c)
        prior[c]= Nc/n
        text[c]=ConcatenateTextOfAllDocsInClass(D,c)
        s=0
        for token in v:
            t[c+'.'+token]=CountTokensOfTerm(text[c],token)
            s=s+CountTokensOfTerm(text[c],token)
        denominator = findSmoothedSum(v,c,t)
        s_sum[c]=denominator
        for token in v:
            condProb[c+'.'+token] = (t[c+'.'+token]+1)/denominator
    return v, prior, condProb, s_sum

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

'''
this function applies multinomial naive bayes to test document located at location = d
'''
def applyMultinomialNB(C,v,prior,condProb,d, s_sum):
    global count_spam_docs
    global count_ham_docs
    data=d
    score={}
    updateFromData= extractTokensFromDoc(v,data,condProb, s_sum)
    w = updateFromData[0]
    smoothedCondProb = updateFromData[1]
    for c in C:
        score[c] = math.log10(prior[c])
        for token in w:
            score[c] = score[c] + math.log10(smoothedCondProb[c+'.'+token])
    # print(score)
    if score['ham']>score['spam']:
        # print("The document belongs to class ham")
        count_ham_docs = count_ham_docs + 1
    else:
        # print("The document belongs to class spam")
        count_spam_docs = count_spam_docs + 1
'''
sub function of testing function called applyMultinomialNB
'''
def extractTokensFromDoc(v,data,condProb, s_sum):
    # result[0] is tokens obtained from splitting data
    vocabLen = len(v)
    vocabData = set(data.split(" "))
    # result[1] is a dictionary containing smoothed cond prob for items having zero frequency in the vocabulary v

    for token in vocabData:
        token_ham= 'ham.'+token
        token_spam = 'spam.'+token
        if token_ham not in condProb.keys():
            condProb[token_ham]= 1/(s_sum['ham'] + vocabLen)
        if token_spam not in condProb.keys():
            condProb[token_spam]= 1/(s_sum['spam'] + vocabLen)

    return vocabData,condProb

'''
main function for testing of documents in a folder given by directory folder given by path
'''
def test_NB(C,res,path_folder):
    global count_ham_docs
    global count_spam_docs
    D= getDocs(path_folder+'/ham', path_folder+'/spam')
    #D[0] is a list of documents in folder ham, D[1] is a list of data of documents in folder spam
    for doc in D[0]:
     applyMultinomialNB(C,res[0],res[1],res[2],doc, res[3])
    print('Accuracy for test set documents in ham folder: ' + str(count_ham_docs/ len(D[0])))
    count_spam_docs = 0
    count_ham_docs = 0
    for doc in D[1]:
      applyMultinomialNB(C, res[0], res[1], res[2],doc, res[3])
    print('Accuracy for test set documents in spam folder: ' + str(count_spam_docs / len(D[1])))

if __name__== '__main__':
    path_train_ham = '../dataSet2/train/ham'
    path_train_spam= '../dataSet2/train/spam'
    count_spam_docs=0
    count_ham_docs=0
    D= getDocs(path_train_ham, path_train_spam)
    C=['ham','spam']
    res=TrainMultinomialNB(C,D)
    # print('=======================conditional_probability=========================')
    # print(res[2])
# res[0]=vocab; res[1]=prior; res[2]=condProb; res[3]=denominator of ham and spam
    path_test_folder = '../dataSet2/test'
    test_NB(C,res,path_test_folder)
    # applyMultinomialNB(C,res[0],res[1],res[2],path_test_doc, res[3])

#    print(res[0])
#    print('=======================prior=========================')
#    print(res[1])

   