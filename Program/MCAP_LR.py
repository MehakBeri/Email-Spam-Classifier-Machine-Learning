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
calculates conditional probability; y=output value (1 for spam and 0 for ham); X= input vector; W=weight vector; returns conditional probability
'''
def condProb(y,X,W,l):
    p=0
    for i in range(len(X[0])):
       p=p+W[i]*X[l][i]
    x=(W[0])+(p)
    # temp= float(decimal.Decimal(2.71828**(W[0]+p)))
    # if y==1:
    #     ans = temp/ (1+temp)
    # else:
    #     ans = 1/(1+temp)
    "Numerically-stable sigmoid function."
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    else:
        # if x is less than zero then z will be small, denom can't be
        # zero because it's 1+z.
        z = math.exp(x)
        return z / (1 + z)
    # return ans

'''
estimates weights based on 70% training data containing spam as well as ham; initialize with a weights=0; step size=n=0.01; modifies vector W after a fixed no. = 50 iterations;
returns a weight vector; L is lambda for regularization
'''
def estimateWeights(ham,spam,L):
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
    n=0.01 #take initial step size as 0.01
    for iteration in range(100):
        for i in range(len(W)):
             sumOverL=0
             # for docs in ham folder
             rows = len(X_ham)
             y = 0
             for j in range(rows):
                 sumOverL = sumOverL + ((X_ham[j][i])*(y-condProb(1,X_ham,W,j)))
             # for docs in spam folder
             rows = len(X_spam)
             y = 1
             for k in range(rows):
                 sumOverL= sumOverL + ((X_spam[k][i])*(y-condProb(1,X_spam,W,k)))
             W[i] = W[i] + (n*sumOverL) - (n*L*W[i])
    return W

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
gets the unique words 
'''
def getVocab(data):
    V=['DummyTextForW[0]'] #since X0 is always one , and the notation starts with words starting X1
    list_data= list(data)
    V.extend(list_data)
    return V

'''
tests accuracy on test set using W and L values along with data
'''
def test(train_ham,train_spam,ham,spam,L):
    print("Value of lambda: "+str(L))
    W = estimateWeights(train_ham, train_spam, L)
    global V
    X_ham = getBoolForDoc(ham)
    X_spam = getBoolForDoc(spam)
    ham_i=0
    spam_i=0
    for i in range(len(X_ham)):
        sum_ham = 0
        for j in range(len(X_ham[0])):
            sum_ham= sum_ham+ (W[j]*X_ham[i][j])
        if sum_ham>0:
            ham_i=ham_i+1
    for ii in range(len(X_spam)):
        sum_spam = 0
        for jj in range(len(X_spam[0])):
            sum_spam= sum_spam+ (W[jj]*X_spam[ii][jj])
        if sum_spam<0:
            spam_i=spam_i+1
    accuracy = ((ham_i+spam_i)/(len(X_ham)+len(X_spam)))*100
    print("Accuracy on given set is: "+str(accuracy)+"%"+"; Accuracy on ham: "+str((ham_i/len(X_ham))) +" ; Accuracy on spam: "+str((spam_i/len(X_spam))))

if __name__== '__main__':
    dataset='3'
    train_ham='../DataSetLR/dataSet'+dataset+'/train/ham70'
    validation_ham='../DataSetLR/dataSet'+dataset+'/train/ham30'
    full_train_ham='../dataSet'+dataset+'/train/ham'
    train_spam='../DataSetLR/dataSet'+dataset+'/train/spam70'
    validation_spam='../DataSetLR/dataSet'+dataset+'/train/spam30'
    full_train_spam='../dataSet'+dataset+'/train/spam'
    test_ham='../DataSetLR/dataSet'+dataset+'/test/ham'
    test_spam='../DataSetLR/dataSet'+dataset+'/test/spam'

    V= []
    # W=estimateWeights(train_ham,train_spam,0)
    # print("Estimating weights without regularization on 70% training data")
    # print(W)
    # print("Testing on 30% validation data")
    # test(train_ham, train_spam, validation_ham, validation_spam, 195)
    # test(train_ham, train_spam, validation_ham, validation_spam, 300)
    # test(train_ham, train_spam, validation_ham, validation_spam, 405)
    # print("Testing on full training set "+dataset+" with a chosen value of lambda as 230")
    # test(full_train_ham,full_train_spam,test_ham,test_spam,230)
    print("Testing on full training set "+dataset+" with a chosen value of lambda as 207")
    test(full_train_ham,full_train_spam,test_ham,test_spam,207)