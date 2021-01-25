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
import keras

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

#If your model achieves > 20% accuracy, you receive full credit for this question.
#Note there may be some variation from submission to submission, 
#but this low threshold is designed so it can be accomplished in 500 iterations
                  
                  
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

    def fit(self, maxIterations, learningRate, batchSize):
            self.params['W1'] = np.random.randn(self.inputSize, self.hiddenSize)
            self.params['W2'] = np.random.randn(self.hiddenSize, self.outputSize)
            
    def predict(self, X):
        hiddenLayerValues = np.dot(X, self.params['W1'])
        scores = hiddenLayerValues.dot(self.params['W2'])
        y_pred = np.argmax(scores, axis=1)
        return(y_pred)


#=========================================
#=========================================
#LAB QUESTION 2
#=========================================
#=========================================
#FUNCTION NAME: forwardTanh
#Parameters: x (arbitrary dimensions, numpy)
#Outputs: out, cache, where cache is the incoming x, 
#and out is the output of the activation layer.

#Example input:
#testInput = np.array([10,20,30,-400])

#The specific line of code I run to check your grade is similar to:
#perCor = np.mean(correctFunctionReturn[0]==studentFunctionReturn[0]) 

def forwardTanh(x):
    out = np.array([442,442,442,442])
    cache = x
    return(out, cache)


#=========================================
#=========================================
#LAB QUESTION 3
#=========================================
#=========================================
#FUNCTION NAME: forwardSigmoid
#Parameters: x (arbitrary dimensions, numpy)
#Outputs: out, cache, where cache is the incoming x, 
#and out is the output of the activation layer.

#Example input:
#testInput = np.array([10,20,30,-400])

#The specific line of code I run to check your grade is similar to:
#perCor = np.mean(correctFunctionReturn[0]==studentFunctionReturn[0]) 

def forwardSigmoid(x):
    out = np.array([442,442,442,442])
    cache = x
    return(out, cache)

#=========================================
#=========================================
#LAB QUESTION 4
#=========================================
#=========================================
#FUNCTION NAME: forwardLeakyRelu
#Parameters: x (arbitrary dimensions, numpy)
#Outputs: out, cache, where cache is the incoming x, 
#and out is the output of the activation layer.

#Example input:
#testInput = np.array([10,20,30,-400])

#The specific line of code I run to check your grade is similar to:
#perCor = np.mean(correctFunctionReturn[0]==studentFunctionReturn[0]) 

def forwardLeakyRelu(x):
    out = np.array([442,442,442,442])
    cache = x
    return(out, cache)


#=========================================
#=========================================
#LAB QUESTION 5
#=========================================
#=========================================
#FUNCTION NAME: backwardLeakyRelu
#Parameters: upstreamGradient and cache (same dimensions; np arrays)
#In this case, the cache holds the "x" value from the forward pass.
#Outputs: array / matrix / tensor of gradients to pass further upstream.

#Example input:
#cache = np.array([10,20,30,-400])
#upstreamGradient = np.array([3.1, .04, -13.3, 104.2])

#The specific line of code I run to check your grade is similar to:
#perCor = np.mean(correctFunctionReturn==studentFunctionReturn) 

def backwardLeakyRelu(upstreamGradient, cache):
    x = cache
    dx = np.array([442,442,442,442])
    return(dx)


#=========================================
#=========================================
#LAB QUESTION 6
#=========================================
#=========================================
#FUNCTION NAME: convolutionalForward
#Parameters: X - shape is (N, Height, Width, Channels)
#            W - shape is (F, filterSize, filterSize, imageChannels)
#            B - shape is (F,)
#Stride should be hard coded to 2.
#No pooling.

#Outputs: One return (out), which is a shape (N, F) output of activation surfaces
#that have been maxpooled (just like the example implementation).  No cache, please.

#Example input:
#exampleInput = X_train[0:10]
#exampleInput.shape outputs (10, 32, 32, 3)

#The specific line of code I run to check your grade is similar to:
#W = np.random.randn(numberOfFilters,i['filterSize'],i['filterSize'], 3)
#B = np.random.randn(i['outputSize'])
#convolutionalForwardOut = submission.convolutionalForward(exampleInput, W, B)

#I then compare your answer to a pre-calculated answer with a set of static weights.  
#If they are equal, you receive credit for this question.

def convolutionalForward(X, W, B, stride=2):
    (N, Height, Width, Channels) = X.shape
    (F, filterSize, filterSize, imageChannels) = W.shape

    out = np.zeros((N,F))

    return(out)


#=========================================
#=========================================
#LAB QUESTION 7
#=========================================
#=========================================
#FUNCTION NAME: submissionNet
#Parameters: None.
#Returns a fully defined keras model 

#After you submit, the model is fit on a node that can run for no more than
#approximately 20 minutes. The calls that are performed are:
#model = submission.submissionNet()
#X_train is shape (40000, 32, 32, 3)
#y_train has been converted to a categorical, and is shape (40000,10)
#model.fit(x=X_train, y=y_train,
#          batch_size=512,
#          epochs=5)
#model.evaluate(X_test, y_test)
#Your grade is based on the categorical accuracy of the model evaluation.
#Note that you will have access to up to 6GB of memory for your run.
#While unlikely to be an issue for this assignment, 
#you can run into OOM (out of memory) errors if you exceed that threshold.

#You *cannot* use predefined networks to accomplish this task; rather, you 
#must define your own here.  To get a 100%, you must hit a baseline
#categorical accuracy of 35% within the 5 epoch limit.  You must achieve
#at least 20% for any credit at all, and your score will scale between
#20% and 35%.

#IMPORTANT: Your model compile must include metrics=['categorical_accuracy']

def submissionNet():
    m = keras.models.Sequential()
    m.add(keras.layers.Conv2D(filters=512,
                              kernel_size=(2,2),
                              activation="tanh",
                              input_shape=(32,32,3)))
    m.add(keras.layers.GlobalAveragePooling2D())
    m.add(keras.layers.Dense(units=10))
    m.compile(optimizer=keras.optimizers.SGD(learning_rate=.001),
                                            loss='categorical_hinge',
                                            metrics=['categorical_accuracy'])
    
    return(m)