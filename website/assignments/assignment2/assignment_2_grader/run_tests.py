import json
import sys
from datetime import datetime
import os
import pickle
import numpy as np
import tarfile
import submission
import requests

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
question["max_score"] = 35
question["name"] = "Two Layer Neural Network Implementation"
question["output"] = ""
question["score"] = 0

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
  print("I am running your model five time.")
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
  if(percentCorrect < 0.20):
    print("Our threshold is 20 percent - so, your implementation receives " + str(points) + " points!")
    question["score"] =  question["score"] + (points/100 * question["max_score"])
    question["output"] = "Model succesfully ran with an average accuracy of " + str(round(percentCorrect*100,2)) + ".  See the log for recommendations on how to improve."
  else:
    question["score"] = question["max_score"]
    question["output"] = "Model succesfully ran with an average accuracy of " + str(round(percentCorrect*100,2))
except Exception as e:
  print("I had an error executing your code!")
  print("\nWhat I tried to run: ")
  print("studentPreds = studentNet.predict(X_test)")
  print("\nERROR I received:")
  print(str(e))
  question["output"] = "Error - see log."

ret["tests"].append(question)
Q1leaderboardScore = str(round(percentCorrect*100,2))

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
    dx[x <= 0] = 0.01
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

json.dumps(ret)
outF = open("/autograder/results/results.json", "w")
json.dump(ret, outF)
outF.close()
