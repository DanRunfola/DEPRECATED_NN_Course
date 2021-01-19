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
#CLASS NAME: twoLayerNet 
#CLASS DESCRIPTION: Implement a two layer neural network
#CLASS FUNCTIONS: In addition to the required __init__, this class should
#have a .fit and .predict capability.

#The grading code will run the following line against your code,
#and your score is determined based on the performance of your
#implementation.  You are free to implement any default
#weight initialization and loss function, but there must be
#a default!

#studentNet = submission.twoLayerNet(inputSize = 3072, 
                  #hiddenSize = 100, 
                  #outputSize = 10, 
                  #X_train=X_train, 
                  #y_train=y_train)
#studentNet.fit(maxIterations=500, learningRate = 1e-7, batchSize = 512)
#studentNet.predict(X_test)

#If your model achieves > 25% accuracy, you receive full credit for this question.
                  
                  
class twoLayerNet():
    def __init__(self, inputSize, hiddenSize, outputSize, 
                X_train, y_train,
                lossType = "svmMulticlass",
                lossParams = {"epsilon": 1},
                weightType = "random"):
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize
        self.params = {}
        self.gradients = {}
        self.lossType = lossType
        self.lossParams = lossParams
        self.weightType = "random"
        self.X_train = X_train
        self.y_train = y_train

        if(self.weightType == "random"):
            self.params['W1'] = np.random.randn(self.inputSize, self.hiddenSize)
            self.params['W2'] = np.random.randn(self.hiddenSize, self.outputSize)

    def fit(self, maxIterations, learningRate, batchSize, visualization=True):
        
        currentIteration = 0
        while currentIteration < maxIterations:
            randomSelection = np.random.randint(len(self.X_train), size=batchSize)
            xBatch = self.X_train[randomSelection,:]
            yBatch = self.y_train[randomSelection]

            self.hiddenLayerValues = np.dot(xBatch, self.params['W1'])
            self.scores = self.hiddenLayerValues.dot(self.params['W2'])
  
            if(self.lossType == "svmMulticlass"):
                e = self.lossParams["epsilon"]
                countSamples = len(xBatch)
                countClasses = self.scores.shape[0]
                trueClassScores = self.scores[np.arange(self.scores.shape[0]), yBatch]
                trueClassMatrix = np.matrix(trueClassScores).T
                self.correct = np.mean(np.equal(trueClassScores, np.amax(self.scores, axis=1)))
                loss_ij = np.maximum(0, self.scores - trueClassMatrix + e)
                loss_ij[np.arange(countSamples), yBatch] = 0
                self.loss_ij = loss_ij
                self.dataLoss = np.sum(np.sum(loss_ij)) / countSamples
            
            gradients = {}
            svmMask = np.zeros(self.loss_ij.shape)
            svmMask[self.loss_ij > 0] = 1
            rowPostiveCount = np.sum(svmMask, axis=1)
            svmMask[np.arange(self.scores.shape[0]), yBatch] = -1 * rowPostiveCount
            self.gradients['W2'] = np.asmatrix(self.hiddenLayerValues).T.dot(svmMask)
            self.gradients["hiddenLayer"] = np.dot(svmMask, self.params['W2'].T)
            self.gradients['W1'] = np.dot(np.asmatrix(xBatch).T, self.gradients["hiddenLayer"])
            self.params['W1'] += -learningRate*self.gradients['W1']
            self.params['W2'] += -learningRate*self.gradients['W2']   
            currentIteration = currentIteration + 1
            
    def predict(self, X):
        hiddenLayerValues = np.dot(X, self.params['W1'])
        scores = hiddenLayerValues.dot(self.params['W2'])
        y_pred = np.argmax(scores, axis=1)
        return(y_pred)                  