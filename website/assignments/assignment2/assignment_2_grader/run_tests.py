import json
import sys
from datetime import datetime
import os
import pickle
import numpy as np
import tarfile
import submission
import requests
import keras

basePath = str(os.path.abspath(__file__))[:-13]

#Top level returns:
ret = {}
ret["output"] = "I have completed my assessment of your submission for lab 1."
ret["visibility"] = "visible"
ret["stdout_visibility"] = "visible"
max_score = 100

#Code Tests:
ret["tests"] = []
score = 0

r = requests.get("https://icss.wm.edu/data442/assets/cifar-10-python.tar.gz")

open(basePath + "/cifar-10-python.tar.gz", 'wb').write(r.content)

#Open test data
tarfile.open(basePath + "/" + "cifar-10-python.tar.gz").extractall("./")
cifar10_dir = 'cifar-10-batches-py'

xs = []
ys = []
for b in range(1,6):
    d = os.path.join(cifar10_dir, 'data_batch_%d' % (b, ))
    
    with open(d, 'rb') as f:
        datadict = pickle.load(f, encoding='latin1')
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
        y = np.array(Y)
    xs.append(X)
    ys.append(y)
X_train = np.concatenate(xs)
y_train = np.concatenate(ys)

with open(os.path.join(cifar10_dir, "test_batch"), 'rb') as f:
    datadict = pickle.load(f, encoding='latin1')
    X = datadict['data']
    Y = datadict['labels']
    X_test = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
    y_test = np.array(Y)

with open('testTrainLab2.pickle', 'wb') as f:
    labData = {}
    labData["X_train"] = X_train
    labData["y_train"] = y_train
    labData["X_test"] = X_test
    labData["y_test"] = y_test
    pickle.dump(labData, f)

with open("testTrainLab2.pickle", "rb") as f:
    labData = pickle.load(f)

X_train = np.reshape(labData["X_train"], (labData["X_train"].shape[0], -1))
X_test = np.reshape(labData["X_test"], (labData["X_test"].shape[0], -1))

y_train = labData["y_train"]
y_test = labData["y_test"]





startTime = datetime.now()

#================================
#================================
#QUESTION 1
#================================
#================================
print("\nCommencing assessment of code submitted for question 1.")
question = {}
question["max_score"] = 15
question["name"] = "Two Layer Neural Network Implementation"
question["output"] = ""
question["score"] = 0
Q1leaderboardScore = 0

try:
  studentNet = submission.twoLayerNet(inputSize = 3072, 
                  hiddenSize =100, 
                  outputSize = 10, 
                  X_train=X_train, 
                  y_train=y_train)
except Exception as e:
  print("I had an error executing your code!")
  print("\nWhat I tried to run: ")
  print("studentNet = submission.twoLayerNet(inputSize = 3072,")
  print("             hiddenSize =100, ")
  print("             X_train=X_train, ")
  print("             y_train=y_train)")
  print("\nERROR I received:")
  print(str(e))
  question["output"] = "Error - see log."

try:
  studentNet.fit(maxIterations=500, learningRate = 1e-7, batchSize = 512)
except Exception as e:
  print("I had an error executing your code!")
  print("\nWhat I tried to run: ")
  print("studentNet.fit(maxIterations=500, learningRate = 1e-7, batchSize = 512)")
  print("\nERROR I received:")
  print(str(e))
  question["output"] = "Error - see log."

try:
  studentPreds = studentNet.predict(X_test)
  print("Model succesfully initialized, fit and predicted a test set.  Technical checks passed.")
  print("Commencing output tests to check accuracy.")


  #Run the model 5 times if everything worked so far.
  #This is to reduce variance in reported scores.
  x = 0
  resultArray = []
  print("I am running your model five times.")
  print("I will record the test accuracy for each run and report it here.")
  print("Your grade will be based on the average of these runs.")
  while x < 5:
    x = x + 1
    studentNet.fit(maxIterations=500, learningRate = 1e-7, batchSize = 512)
    studentPreds = studentNet.predict(X_test)
    percentCorrectIt = np.mean(np.equal(y_test, studentPreds))
    print("Test " + str(x) + " accuracy: " + str(round(percentCorrectIt*100,2)) + " %.")
    resultArray.append(percentCorrectIt)
  
  percentCorrect = np.mean(percentCorrectIt)
  
  points = min(1, percentCorrect/0.20)*100
  print("Average Test Dataset Accuracy: " + str(round(percentCorrect*100,2)) + " %")
  if(percentCorrect < 0.15):
    print("You must achieve at least 15% model accuracy for any credit on this question.")
    question["output"] = "Your model ran, but with an accuracy too low to be awarded any points.  15 percent is the minimum accepted."
  elif(percentCorrect < 0.20):
    print("Our threshold is 20 percent - so, your implementation receives " + str(points) + " points!")
    question["score"] =  question["score"] + (points/100 * question["max_score"])
    question["output"] = "Model succesfully ran with an average accuracy of " + str(round(percentCorrect*100,2)) + ".  See the log for recommendations on how to improve."
    Q1leaderboardScore = str(round(percentCorrect*100,2))
  else:
    question["score"] = question["max_score"]
    question["output"] = "Model succesfully ran with an average accuracy of " + str(round(percentCorrect*100,2))
    Q1leaderboardScore = str(round(percentCorrect*100,2))
except Exception as e:
  print("I had an error executing your code!")
  print("\nWhat I tried to run: ")
  print("studentPreds = studentNet.predict(X_test)")
  print("\nERROR I received:")
  print(str(e))
  question["output"] = "Error - see log."

ret["tests"].append(question)


#================================
#================================
#QUESTION 2
#================================
#================================
print("\nCommencing assessment of code submitted for question 2.")
question = {}
question["max_score"] = 5
question["name"] = "tanh activation"
question["output"] = ""
question["score"] = 0

def forwardTanh(x):
    out = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    cache = x
    return(out, cache)

testArray = np.array([0.5, 0.5, 0.25, -0.25, 100, 10, -10, -1, 0])
correctOut = forwardTanh(testArray)

print("I am comparing your tanh activation function to a correctly implemented one.")
try:
  studentOut = submission.forwardTanh(testArray)
  perCor = np.mean(correctOut[0]==studentOut[0]) 
  points = perCor * question["max_score"]
  question["output"] = str(round(perCor, 2)*100) + " percent of my test entries matched yours.  Points awarded: " + str(points)
  print(str(round(perCor, 2) * 100) + " percent of my test entries matched yours.  Points awarded: " + str(points))
  question['score'] = points
except Exception as e:
  print("I was unable to call your forwardTanh function.  Here is the error I received: " + str(e))
  question["output"] = "Error running your function.  Check the log."

ret["tests"].append(question)

#================================
#================================
#QUESTION 3
#================================
#================================
print("\nCommencing assessment of code submitted for question 3.")
question = {}
question["max_score"] = 5
question["name"] = "sigmoid activation"
question["output"] = ""
question["score"] = 0

def forwardSigmoid(x):
    out = 1/(1+np.exp(-x))
    cache = x
    return(out, cache)
    
testArray = np.array([0.5, 0.5, 0.25, -0.25, 100, 10, -10, -1, 0])
correctOut = forwardSigmoid(testArray)

print("I am comparing your tanh activation function to a correctly implemented one.")
try:
  studentOut = submission.forwardSigmoid(testArray)
  perCor = np.mean(correctOut[0]==studentOut[0]) 
  points = perCor * question["max_score"]
  question["output"] = str(round(perCor, 2)*100) + " percent of my test entries matched yours.  Points awarded: " + str(points)
  print(str(round(perCor, 2)* 100) + " percent of my test entries matched yours.  Points awarded: " + str(points))
  question['score'] = points
except Exception as e:
  print("I was unable to call your forwardSigmoid function.  Here is the error I received: " + str(e))
  question["output"] = "Error running your function.  Check the log."

ret["tests"].append(question)

#================================
#================================
#QUESTION 4
#================================
#================================
print("\nCommencing assessment of code submitted for question 4.")
question = {}
question["max_score"] = 5
question["name"] = "Leaky relu forward"
question["output"] = ""
question["score"] = 0

def forwardLeakyRelu(x):
    out = np.maximum(x, 0.01 * x)
    cache = x
    return(out, cache)
    
testArray = np.array([0.5, 0.5, 0.25, -0.25, 100, 10, -10, -1, 0])
correctOut = forwardLeakyRelu(testArray)

print("I am comparing your leaky relu activation function to a correctly implemented one.")
try:
  studentOut = submission.forwardLeakyRelu(testArray)
  perCor = np.mean(correctOut[0]==studentOut[0]) 
  points = perCor * question["max_score"]
  question["output"] = str(round(perCor, 2)*100) + " percent of my test entries matched yours.  Points awarded: " + str(points)
  print(str(round(perCor, 2)* 100) + " percent of my test entries matched yours.  Points awarded: " + str(points))
  question['score'] = points
except Exception as e:
  print("I was unable to call your forwardLeakyRelu function.  Here is the error I received: " + str(e))
  question["output"] = "Error running your function.  Check the log."

ret["tests"].append(question)

#================================
#================================
#QUESTION 5
#================================
#================================
print("\nCommencing assessment of code submitted for question 5.")
question = {}
question["max_score"] = 10
question["name"] = "Leaky relu backward"
question["output"] = ""
question["score"] = 0

def backwardLeakyRelu(upstreamGradient, cache):
    x = cache
    dx = np.array(upstreamGradient, copy=True)
    dx[x <= 0] = 0.01 * x
    return(dx)
    
testArray = np.array([0.5, 0.5, 0.25, -0.25, 100, 10, -10, -1, 0])
upstreamGradientTest = np.array([0.05, -0.045, -0.25, 0.250, 1.45, 0.64, -1, -1, 0])

correctOut = backwardLeakyRelu(upstreamGradientTest, testArray)

print("I am comparing your backward leaky relu function to a correctly implemented one.")
try:
  studentOut = submission.backwardLeakyRelu(upstreamGradientTest, testArray)
  perCor = np.mean(correctOut==studentOut) 
  points = perCor * question["max_score"]
  question["output"] = str(round(perCor, 2)*100) + " percent of my test entries matched yours.  Points awarded: " + str(points)
  print(str(round(perCor, 2)* 100) + " percent of my test entries matched yours.  Points awarded: " + str(points))
  question['score'] = points
except Exception as e:
  print("I was unable to call your backwardLeakyRelu function.  Here is the error I received: " + str(e))
  question["output"] = "Error running your function.  Check the log."

ret["tests"].append(question)

#================================
#================================
#QUESTION 6
#================================
#================================
print("\nCommencing assessment of code submitted for question 6.")
question = {}
question["max_score"] = 10
question["name"] = "Convolutional Forward"
question["output"] = ""
question["score"] = 0

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

exampleInput = labData["X_train"][345:350]
W = np.random.randn(2,5,5,3)
B = np.random.randn(2)
correctOut = convolutionalForward(X=exampleInput, W=W, B=B, stride=2)

print("I am comparing your forward convolutional function to a correctly implemented one.")
try:
  studentOut = submission.convolutionalForward(X=exampleInput, W=W, B=B, stride=2)
  perCor = np.mean(correctOut==studentOut) 
  points = perCor * question["max_score"]
  question["output"] = str(round(perCor, 2)*100) + " percent of my test entries matched yours.  Points awarded: " + str(points)
  print(str(round(perCor, 2)* 100) + " percent of my test entries matched yours.  Points awarded: " + str(points))
  question['score'] = points
except Exception as e:
  print("I was unable to call your convolutional forward function.  Here is the error I received: " + str(e))
  question["output"] = "Error running your function.  Check the log."

ret["tests"].append(question)


#================================
#================================
#QUESTION 7
#================================
#================================
print("\nCommencing assessment of code submitted for question 7.")
question = {}
question["max_score"] = 20
question["name"] = "Full Network Implementation"
question["output"] = ""
question["score"] = 0

from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

Q7leaderboardScore = 0

print("I am loading your model...")
try:
  model = submission.submissionNet()
  print(model.summary())

  print("Model succesfully loaded.  Fitting.")
  try:
    model.fit(x=labData['X_train'], y=y_train,
              batch_size=64,
              epochs=5,
              verbose=2,
              use_multiprocessing=True)

    print("Model succesfully fit.  Running the final evaluation.")
    try:
      out = model.evaluate(labData['X_test'], y_test, return_dict=True, verbose=0)
      print("Your model categorical accuracy was " + str(out["categorical_accuracy"]) + ".")
      Q7leaderboardScore = str(round(out["categorical_accuracy"]*100,2))
      if(out["categorical_accuracy"] > 0.35):
        question["score"] = question["max_score"]
      else:
        question["score"] = (max(out["categorical_accuracy"]-0.2,0) / 0.15) * question["max_score"]
        question["output"] = "To get any points you must get a minimum of 20 percent accuracy on the test set of data, and if you achieve a 35% you will receive a 100.  Your score will scale for points over 20 percent but under 35 percent.  Your accuracy was " + str(round(out["categorical_accuracy"]*100,2))
    except Exception as e:
      print("Your model evaluation failed.  Here is the error I received: " + str(e))
      question["output"] = "Error running your function.  Check the log."


  except Exception as e:
    print("Your model failed to fit.  Here is the error I received: " + str(e))
    question["output"] = "Error running your function.  Check the log."


except Exception as e:
  print("I was unable to call your network.  Here is the error I received: " + str(e))
  question["output"] = "Error running your function.  Check the log."

ret["tests"].append(question)


#================================
#================================
#QUESTION 8
#================================
#================================
#https://www.tensorflow.org/datasets/catalog/resisc45
#Note - 
#Agriculture is a straight copy, as is
#buildings
print("\nCommencing assessment of code submitted for question 8.")
question = {}
question["max_score"] = 30
question["name"] = "Building a Model - Real World Case"
question["output"] = ""
question["score"] = 0
mAcc = 0

import pathlib

try:
  print("Loading Model...")
  studentModel = keras.models.load_model("/autograder/submission/Q8.h5")
  dataGenerator = keras.preprocessing.image.ImageDataGenerator()
  test = dataGenerator.flow_from_directory(basePath + "/testImages", class_mode='categorical', batch_size=64)

  print("Testing Model based on independent test set...")
  try:
    modelOutcome = studentModel.evaluate(test)
    print("Your model achieved an accuracy of " + str(round(modelOutcome[1]*100,4)) + " percent.")
    print("This is relative to the goal of 50 percent.")
    mAcc = modelOutcome[1] * 100
    question["score"] = min(1, mAcc / 50) * question["max_score"]

    print("Your score for this question is currently " + str(question["score"]))
    
  except Exception as e:
    print("I was unable to run your model on my test dataset.")
    print("Exception: " + str(e))
    question["output"] = "Something went wrong!  Check the log."


except Exception as e:
  print("I was unable to load Q8.h5.  Please check your upload is correctly formatted.")
  print("Note that if you have not yet started on Question 8, you may see this error.")
  print("(i.e., if you have not yet started submitting a Q8.h5 file!)")
  print("Exception: " + str(e))
  question["output"] = "Something went wrong!  Check the log."

ret["tests"].append(question)

#LEADERBOARD
ret["leaderboard"] = []

tim = {}
tim["name"] = "Runtime (seconds)"
tim["value"] = str(datetime.now() - startTime)
tim["order"] = "asc"
ret["leaderboard"].append(tim)

acc = {}
acc["name"] = "Accuracy (Percentage) of Q1 Model"
acc["value"] = Q1leaderboardScore
ret["leaderboard"].append(acc)

acc = {}
acc["name"] = "Accuracy (Percentage) of Q7 Model"
acc["value"] = Q7leaderboardScore
ret["leaderboard"].append(acc)

acc = {}
acc["name"] = "Accuracy (Percentage) of Q8 Model"
acc["value"] = mAcc
ret["leaderboard"].append(acc)

json.dumps(ret)
outF = open("/autograder/results/results.json", "w")
json.dump(ret, outF)
outF.close()
