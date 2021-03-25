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
    out = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
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
    out = 1/(1+np.exp(-x))
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
    out = np.maximum(x, 0.01 * x)
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
    dx = np.array(upstreamGradient, copy=True)
    dx[x <= 0] = 0.01 * x
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
    convH = int((Height-filterSize)/stride)+1
    convW = int((Width-filterSize)/stride)+1
    activationSurface = np.zeros((N, F, convH, convW, imageChannels))
    out = np.zeros((N,F))

    for i in range(N):
        x = X[i]
        for f in range(F):
            for h in range(0, Height, stride):
                for w in range(Width):
                    for c in range(Channels):
                        y_upper = h * stride
                        y_lower = y_upper + filterSize
                        x_left = w * stride
                        x_right = x_left + filterSize
                        window = x[y_upper:y_lower, x_left:x_right, :]
                        if((window.shape[0] == filterSize) and window.shape[1] == filterSize):
                            s = np.multiply(window, W[f])
                            activationSurface[i,f,h,w,c] = np.sum(s)
                            activationSurface[i,f,h,w,c] = activationSurface[i,f,h,w,c] + np.sum(B)
                            out[i,f] = np.max(activationSurface[i,f,h,w,c])
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
#Note that you will have access to up to 6GB of memory for your run,
#which further limits the number of parameters and types of models you can
#create.  

#***The primary challenge for this question is to build a model which is both
#small enough to train on limited infrastructure, but powerful enough to 
#get a somewhat-reasonable accuracy.  I expect this will require a lot of
#trial-and-error; don't be surprised if something that runs on your local
#computer will not run on the cloud.***

#You *cannot* use predefined networks to accomplish this task; rather, you 
#must define your own here.  To get a 100%, you must hit a baseline
#categorical accuracy of 35% within the 5 epoch limit.  You must achieve
#at least 20% for any credit at all.

#IMPORTANT: Your model compile must include metrics=['categorical_accuracy']

def submissionNet():
    #######FOR KEY:
    ####THIS IS A DUMMY MODEL SLOTTED IN TO MAKE THIS RUN QUICKLY FOR TESTING.
    ####THE "KEY" MODEL IS COMMENTED BELOW.

    #m = keras.models.Sequential()
    #m.add(keras.layers.GlobalAveragePooling2D())
    #m.add(keras.layers.Dense(units=10))
    #m.compile(optimizer=keras.optimizers.SGD(learning_rate=.001),
    #                                        loss='categorical_hinge',
    #                                        metrics=['categorical_accuracy'])
    
    m = keras.models.Sequential()
    m.add(keras.layers.Conv2D(filters=64,
                              kernel_size=(2,2),
                              activation="tanh",
                              input_shape=(32,32,3)))
    m.add(keras.layers.BatchNormalization())
    m.add(keras.layers.Conv2D(filters=64,
                            kernel_size=(4,4),
                            activation="tanh",
                            input_shape=(32,32,3)))
    m.add(keras.layers.BatchNormalization())
    m.add(keras.layers.GlobalAveragePooling2D())
    m.add(keras.layers.Dense(units=10))
    m.compile(optimizer=keras.optimizers.SGD(learning_rate=.001),
                                            loss='categorical_hinge',
                                            metrics=['categorical_accuracy'])
    
    return(m)


#=========================================
#=========================================
#LAB QUESTION 8
#=========================================
#=========================================
#In this question, you are going to write
#and fit the best network you can to solve a problem.
#You will then save your model, and submit
#a file called "submission.zip" to gradescope.
#This file must include both (a) a file named
#"Q8.h5", and (b) your submission.py.
#Your grade for this question will be based on
#the accuracy of your saved model.

#To write this model, you must use keras, as
#per the below example.  We will be using
#the UC Merced Land Use database.
#You will find the images in the folder "mercerImages"
#In this assignment.

#Once you fit your model and upload it, it will
#be tested against a completely distinct
#set of images from another dataset you do not
#have access to.  The accuracy of your model in predicting
#this second dataset is what will determine
#your score on this question.

#Note it is totally up to you to subdivide your images
#into test/train sets.  In this example, I only use a training
#dataset.  This is obviously wrong, don't do this!

#Note you don't actually need to submit your submissionNet
#Code - your grade on this question is just based on the Q8.h5 file
#you submit.

#Note you *must* include metrics=['categorical_accuracy'] in your
#modile compilation (i.e., see below).

#If you achieve 50% accuracy on the independent test set,
#you will receive a 100% on this question.

from keras.applications.resnet50 import ResNet50

def q8ExampleNet():
    m = keras.models.Sequential()
    m.add(keras.applications.ResNet50V2(
        weights="imagenet",
        include_top=False,
        pooling="max"
    ))
    m.add(keras.layers.Dense(units=512))
    m.add(keras.layers.Dense(units=256))
    m.add(keras.layers.Dense(units=21))
    
    m.compile(optimizer=keras.optimizers.SGD(learning_rate=.001),
                                            loss='categorical_hinge',
                                            metrics=['categorical_accuracy'])
    
    return(m)

if __name__ == "__main__":
    #Load the images to fit with
    dataGenerator = keras.preprocessing.image.ImageDataGenerator()

    #Note that class_mode = categorical may be different for you, depending on how you
    #develop your model.  In this case, I'm loading the data from folders, and using the
    #folder names as the classes.
    train = dataGenerator.flow_from_directory("website/assignments/assignment2/mercerImages/", 
    class_mode='categorical', 
    batch_size=64, target_size=(224,224))
    
    model = q8ExampleNet()
    model.summary()
    
    model.fit(train, epochs=3, verbose=2)

    #IMPORTANT: Your model must be saved as "Q8.h5"
    model.save("/home/dan/git/D442/website/assignments/assignment2/assignment_2_grader/Q8.h5")

