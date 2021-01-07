#You will submit this python file to receive your grade.
#The code you provide here, and your solutions, will be
#the basis for your grade in this lab.
#The names of functions the algorithm expects are noted throughout this
#file.

#Before you get started, it is highly recommended you submit this
#file as-is to see how it works (of course, you will get a 0 on that submission)!
#You can resubmit as many times as you want up to the deadline.
#I also *highly* suggest you submit after each question.

#Modules for use in this lab.
#Note you cannot use additional modules, 
#unless they are standard libraries (i.e., "os").
import tarfile
import requests
import os
import numpy as np
import pickle

#Note that functions must ALWAYS be named exactly as noted, or you will not be 
#awarded any points.  

#=========================================
#=========================================
#LAB QUESTION 1
#=========================================
#=========================================
#FUNCTION NAME: dataDownload 
#FUNCTION DESCRIPTION: Function to download CIFAR10 data.
#FUNCTION OUTPUT: Path, as a string, of the location CIFAR10 was extracted to.
#FUNCTION NOTES: 
#1) Download the file https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
#2) Extract the file to a path (i.e., "./images")
#3) Return the path that the files were extracted to

#The algorithm that grades this question will check if the CIFAR data is present
#in the path you return.  

def dataDownload(url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
                 outPath = "./images/"):
    
    ###YOUR CODE HERE###

    return(outPath)


#=========================================
#=========================================
#LAB QUESTION 2
#=========================================
#=========================================
#FUNCTION NAME: dataSplit
#FUNCTION DESCRIPTION: Function to ingest and split the CIFAR data
#FUNCTION OUTPUT: Path, as a string, to the pickle containing training and test data.
#FUNCTION NOTES: 
#An example implementation of this function can be found in the
#downloadLabData notebook.  It is highly recommended you use that as your base
#for implementation.

#1) Load each pickled data_batch in the CIFAR dataset.
#2) Reshape and transpose the training data (i.e., 10k observations, 3 bands, 32x32 dimensions)
#3) Repeat this for your test data
#4) Save and pickle all of your results.
#Note you must create a dictionary variable for your pickle.
#The final format of the variable should look like:
#labData = {}
#labData["X_train"] = X_train
#labData["y_train"] = y_train
#labData["X_test"] = X_test
#labData["y_test"] = y_test
#pickle.dump(labData, f)

#The algorithm that grades this question will load your pickled file,
#check that the variable saved is a dictionary,
#check that it has X_train, X_test, y_train and y_test objects,
#and that these objects are representative of the expected CIFAR data.

#Note that this function builds on the last one - i.e., wherever you extracted
#your images to will need to be specified here.  For this example,
#The first function extracted CIFAR to ./images/.

def dataSplit(basePath = "./images/",
              picklePath = "./testTrainLab1.pickle"):
    
    ####YOUR CODE HERE
    ####
    ####END YOUR CODE

    #Pickle your data - leave this as is.  Your code should provide
    #the correct X_train, y_train, X_test, and y_test.
    with open(picklePath, 'wb') as f:
        labData = {}
        labData["X_train"] = X_train
        labData["y_train"] = y_train
        labData["X_test"] = X_test
        labData["y_test"] = y_test
        pickle.dump(labData, f)
    
    return(picklePath)


#=========================================
#=========================================
#LAB QUESTIONS 3, 4, 5
#=========================================
#=========================================
#For questions 3, 4 and 5, reference the below class "NearestNeighbor".

#Q3: Implicit in this class is a definition of 'k' for the 
#k-nearest-neighbors algorithm.  What is it? Your answer
#should be an integer.
def questionThree():
    return(42)
    
#Q4: What is the distance metric implemented?  Choose one of the below.
def questionFour():
    return("Blank")
    #return("Euclidean")
    #return("L1")
    #return("L2")
    #return("L3")
    #return("None of the Above")

#Q5: If you run the below class with the first 100 test cases in 
#the CIFAR dataset, approximately
#what accuracy do you receive?  Choose one of the below.
def questionFive():
    return("Blank")
    #return("10%")
    #return("30%")
    #return("50%")
    #return("70%")
    #return("90%")

class NearestNeighbor:
    def __init__(self):
        pass

    def train(self, X, y):
        self.Xtr = X
        self.ytr = y

    def predict(self, X):        
        Ypred = np.zeros(len(X), dtype=np.dtype(self.ytr.dtype))

        for i in range(0, len(X)):
            l1Distances = np.sum(np.abs(self.Xtr - X[i]), axis=1)
            minimumDistance = np.argmin(l1Distances)
            Ypred[i] = self.ytr[minimumDistance]
        
        return Ypred


#=========================================
#=========================================
#LAB QUESTION 6
#=========================================
#=========================================
#CLASS NAME: knnClassifier
#FUNCTION DESCRIPTION: Function to classify an image based on a knn Classifer with a L2 norm.
#FUNCTION OUTPUT: Class prediction for a list of images, between 0 and 9.
#FUNCTION NOTES: 
#An example implementation of this function can be found in the
#knn notebook.  It is highly recommended you use this as a model.
#The equation for a L2 norm can be found in the lecture.
#You will be graded based on the agreement between your classifier
#and the results a correctly implemented L2 norm k=10 classifier
#would give for 100 test cases.

class knnClassifier:
    def __init__(self):
        pass

    def train(self, X, y):
        self.Xtr = X
        self.ytr = y
    
    def predict(self, X, k):
        Ypred = np.zeros(len(X), dtype=np.dtype(self.ytr.dtype))

        ######CODE IS MISSING HERE!
        #Replace the below for loop (which randomly assigns a class)
        #with your knn!
        for i in range(0, len(X)):
            Ypred[i] = np.random.randint(0,9,1)[0]
        
        ###END OF CODE YOU NEED TO ADD
        
        return Ypred


#=========================================
#=========================================
#LAB QUESTION 7
#=========================================
#=========================================
#FUNCTION NAME: crossFoldValidation
#FUNCTION DESCRIPTION: Function to conduct a cross fold validation on a given classifier
#                      for a single set of hyperparameters.
#REQUIRED FUNCTION PARAMETERS:
#                      modelToValidate - the model to validate, i.e. knnClassifier()
#                      X_train - the full (unsplit) X training dataset
#                      y_train - the full (unsplit) y training dataset
#                      folds - an integer representing the number of folds
#FUNCTION OUTPUT: List of accuracy values - one per fold.
#FUNCTION NOTES: 
#An example implementation of this function can be found in the
#hyperparameters notebook.  It is highly recommended you use this as a model.
#You will be graded based on the agreement between your crossfold
#validation model, and the results of a crossfold validation
#prebuilt on a sample of 1000 observations from the CIFAR10 dataset.
#The value of k = 5 should be *hard coded* for this question.

def crossFoldValidation(modelToValidate,
                        X_train,
                        y_train,
                        folds=5):
    
    k = 5

    ####YOUR CODE GOES HERE.
    #RIGHT NOW THIS IS A FOR LOOP THAT GENERATES A RANDOM ACCURACY.

    accuracies = []

    for i in range(0,folds): 
        accuracies.append(np.random.randint(0,1,1)[0])

    #####END OF BLOCK YOU NEED TO ADD

    return(accuracies)  

#=========================================
#=========================================
#LAB QUESTION 8
#=========================================
#=========================================
#Function Name: svmClassifier
#FUNCTION DESCRIPTION: Function to solve for the multiclass SVM data loss (with regularization).
#REQUIRED FUNCTION PARAMETERS:
#                      X - the full (unsplit) X training dataset
#                      y - the full (unsplit) y training dataset
#                      W - a vector of weights (i.e., W = np.random.randn(3072, 10) * 0.0001)
#                      e - epsilon term for SVM loss
#                      l - Lambda for regularization loss
#FUNCTION OUTPUT: Library with 'dataLoss', 'regLoss', 'totalLoss', and loss matrix loss_ij.

def svmClassifier(X, y, W, e, l):
    scores = X.dot(W)
    countTrainSamples = scores.shape[0]
    countClasses = scores.shape[1] 
    trueClassScores = scores[np.arange(scores.shape[0]), y]
    trueClassMatrix = np.matrix(trueClassScores).T 
    loss_ij = np.maximum(0, (scores - trueClassMatrix) + e) 
    loss_ij[np.arange(countTrainSamples), y] = 0

    ###BELOW CODE NEEDS TO BE MODIFIED TO ACCURATELY CALCULATE DATALOSS, REGLOSS, AND TOTALLOSS.
    #KEEP THE VARIABLE NAMES THE SAME - YOU SHOULD ONLY NEED TO MODIFY THESE THREE LINES.
    dataLoss = np.random.randint(0,100,1)[0]
    regLoss = np.random.randint(0,100,1)[0]
    totalLoss = np.random.randint(0,100,1)[0]
    #END OF BLOCK YOU SHOULD MODIFY
    
    
    return({'dataLoss':dataLoss, 'regLoss':regLoss, 'totalLoss':totalLoss, 'loss_ij':loss_ij})

#=========================================
#LAB QUESTION 9
#=========================================
#=========================================
#Function Name: svmOptimizer
#FUNCTION DESCRIPTION: Function to optimize parameters W in the function svmClassifier
#FUNCTION OUTPUT: Best identified set of parameters W for input svm model (3072 * 10).
#You will be graded based on how close your optimal solution is to the 
#optimal solution the autograder has found.  The autograder uses e=1, l=1, 
#learningRate = .0000001, and 1000 iterations; these can be hard coded in your function.
#You must use your own svmClassifier
#from the previous question as the input model to solve for. 
#The autograder will call this function using the code:
#studentWeights = svmOptimizer(X = X_train, y = y_train)

def svmOptimizer(X, y, model = svmClassifier):
    W = np.random.randn(3072, 10) * .0001
    e = 1
    l = 1
    currentIteration = 1
    maxIterations = 1000
    learningRate = .0000001

    while currentIteration < maxIterations:
        
        #ANALYTIC CODE TO SOLVE FOR W HERE!  Default code
        #is just a random set of guesses, which will be.. bad.
        W = W * np.random.randn(3072, 10) 
        currentIteration = currentIteration + 1
        #END CODE YOU ADD.

    return(W)


#=========================================
#Function tests for your convenience.
#=========================================
if __name__ == '__main__':
  print(dataDownload())
  print(dataSplit())

  with open("testTrainLab1.pickle", "rb") as f:
    labData = pickle.load(f)
    X_train = np.reshape(labData["X_train"], (labData["X_train"].shape[0], -1))
    X_test = np.reshape(labData["X_test"], (labData["X_test"].shape[0], -1))
    y_train = labData["y_train"]
    y_test = labData["y_test"]

    #Cutting down the sample for speed - as noted
    #a few times now, KNN is very slow!
    y_test = y_test[:250]
    X_test = X_test[:250]
    y_train = y_train[:250]
    X_train = X_train[:250]

  print(crossFoldValidation(folds=5,
                        modelToValidate = knnClassifier(),
                        X_train = X_train,
                        y_train = y_train))

  W = np.random.randn(3072, 10) * 0.0001
  print(svmClassifier(X_train, y_train, W, e=1, l=1))

  print(svmOptimizer(X=X_train, y = y_train))
