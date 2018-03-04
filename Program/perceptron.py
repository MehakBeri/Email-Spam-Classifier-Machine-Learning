import os
import math
'''
getDocs function gets the documents inside the directory specified by Path
'''
def getDocs(Path):
    directory = os.listdir(Path)
    files = []
    for file in directory:
        data = getText(Path + '/' + file)
        files.append(data)
    return files

'''
getText functions gets the text included in a file specified by the given filename
'''
def getText(filename):
    with open(filename, 'r', encoding="Latin-1") as myfile:
        data=myfile.read().replace('\n', ' ')
    return data

'''
ham=0; spam=1
here ham=-1, spam=1
estimates weights based on 70% training data containing spam as well as ham; initialize with a weights=0; step size=n=0.1; modifies vector W after a fixed no. = 50 iterations;
returns a weight vector; 
'''
def estimateWeights(ham,spam,N):
    global V
    ham_data=getDocs(ham)
    ham_set= set(ham_data[0].split(" "))
    spam_data=getDocs(spam)
    spam_set= set(spam_data[0].split(" "))
    data= ham_set.union(spam_set)
    V = getVocab(data) #list of unique words in traning set=ham n spam
    W = [0 for i in range(len(V))] #corresponding weights, beginning with initial weights of zero
    X_ham = getBoolForDoc(ham)
    X_spam = getBoolForDoc(spam)
    n=0.1 #take initial step size as 0.01
    for iteration in range(N):
             # for docs in ham folder
             rows = len(X_ham)
             t = 0
             for j in range(rows):
                 o = computeOutput(W,X_ham[j])
                 for wt in range(len(W)):
                     W[wt]=W[wt]+(n*(t-o)*X_ham[j][wt])

             # for docs in spam folder
             rows = len(X_spam)
             t = 1
             for k in range(rows):
                 o = computeOutput(W, X_spam[k])
                 for wt in range(len(W)):
                     W[wt] = W[wt] + (n * (t - o) * X_spam[k][wt])
    return W

'''
computes output given weight vector and input vector; is sum>0=>return 1=spam
'''
def computeOutput(W,X):
    s=W[0]
    for i in range(1,len(W)):
        s=s+(W[i]*X[i])
    if s>0:
        return 1
    else:
        return 0

'''
gets the unique words 
'''
def getVocab(data):
    V=['DummyTextForW[0]'] #since X0 is always one , and the notation starts with words starting X1
    list_data= list(data)
    V.extend(list_data)
    return V
'''
gets boolean value = whether a word is present in that text document or not, for the given document and vocabulary
'''
def getBoolForDoc(Path):
    global V
    folder = os.listdir(Path)
    X=[]
    for doc in folder:
        freq=[1] #frequency for W[0] is X[0] is 1
        data = getText(Path + '/' + doc)
        data_Set = set(data.split(" "))
        for i in range(1,len(V)):
            if V[i] in data_Set:
                freq.append(1)
            else:
                freq.append(0)
        X.append(freq)
    return X

'''
tests accuracy on test set using W and L values along with data
'''
def test(train_ham,train_spam,ham,spam,N):
    print("Value of iteration: "+str(N))
    W = estimateWeights(train_ham, train_spam, N)
    global V
    X_ham = getBoolForDoc(ham)
    X_spam = getBoolForDoc(spam)
    ham_i=0
    spam_i=0
    for i in range(len(X_ham)):
        sum_ham = 0
        for j in range(len(X_ham[0])):
            sum_ham= sum_ham+ (W[j]*X_ham[i][j])
        if sum_ham<0:
            ham_i=ham_i+1
    for ii in range(len(X_spam)):
        sum_spam = 0
        for jj in range(len(X_spam[0])):
            sum_spam= sum_spam+ (W[jj]*X_spam[ii][jj])
        if sum_spam>0:
            spam_i=spam_i+1
    accuracy = ((ham_i+spam_i)/(len(X_ham)+len(X_spam)))*100
    print("Accuracy on given set is: "+str(accuracy)+"%"+"; Accuracy on ham: "+str((ham_i/len(X_ham))) +" ; Accuracy on spam: "+str((spam_i/len(X_spam))))

if __name__== '__main__':
    dataset='3'
    V=[]
    train_ham='../DataSetLR/dataSet'+dataset+'/train/ham70'
    validation_ham='../DataSetLR/dataSet'+dataset+'/train/ham30'
    full_train_ham='../dataSet'+dataset+'/train/ham'
    train_spam='../DataSetLR/dataSet'+dataset+'/train/spam70'
    validation_spam='../DataSetLR/dataSet'+dataset+'/train/spam30'
    full_train_spam='../dataSet'+dataset+'/train/spam'
    test_ham='../DataSetLR/dataSet'+dataset+'/test/ham'
    test_spam='../DataSetLR/dataSet'+dataset+'/test/spam'
    n=151
    # print("Testing on validation training set "+dataset+" with a chosen value of iterations as "+str(n))
    # test(train_ham,train_spam,validation_ham,validation_spam,n)
    print("Testing on full training set "+dataset+" with a chosen value of iterations as "+str(n))
    test(full_train_ham,full_train_spam,test_ham,test_spam,n)