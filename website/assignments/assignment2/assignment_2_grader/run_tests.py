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
question["max_score"] = 20
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

  #Check the total percent accuracy
  percentCorrect = np.mean(np.equal(y_test,studentPreds))
  points = min(1, percentCorrect/0.25)*100
  print("Test Dataset Accuracy: " + str(round(percentCorrect*100,2)) + " %")
  print("Our threshold is 25 percent - so, your implementation receives " + str(points) + " points!")
  print("If you received less than a 100%, consider adding some image preprocessing steps to your class, or...")
  print("trying different weights initialization strategies.  A correctly implemented random initialization with no")
  print("image preprocessing should give you over 90 percent, but the random weights initialization")
  print("will cause your score to vary across submissions.  Keep that in mind as you resubmit (and remember, ")
  print("you can resubmit as much as you want!)")
  question["score"] =  question["score"] + (points/100 * question["max_score"])
  question["output"] = "Model succesfully ran with accuracy of " + str(round(percentCorrect*100,2))
except Exception as e:
  print("I had an error executing your code!")
  print("\nWhat I tried to run: ")
  print("studentPreds = studentNet.predict(X_test)")
  print("\nERROR I received:")
  print(str(e))
  question["output"] = "Error - see log."

ret["tests"].append(question)
Q1leaderboardScore = str(round(percentCorrect*100,2))


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
