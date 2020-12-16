import json
import sys
from datetime import datetime
import os
import pickle
import numpy as np

import submission

#Top level returns:
ret = {}
ret["output"] = "I have completed my assessment of your submission for lab 1."
ret["visibility"] = "visible"
ret["stdout_visibility"] = "visible"
max_score = 100

#Code Tests:
ret["tests"] = []
score = 0

startTime = datetime.now()

#================================
#================================
#QUESTION 1
#================================
#================================
print("\nCommencing assessment of code submitted for question 1.")
question = {}
question["max_score"] = 5
question["name"] = "Test: Code succesfully downloads CIFAR-10"
question["output"] = ""
question["score"] = 0

try:
  outPath = submission.dataDownload()
  print("I am checking what files exist in " + outPath + ": ")

  allPaths = []
  for root, dirs, files in os.walk(outPath):
      path = root.split(os.sep)
      allPaths.append(os.path.basename(root))
      for file in files:
          allPaths.append(file)

  print("Files found: ")
  print(allPaths)

  if(("batches.meta" in allPaths) and 
     ("data_batch_1" in allPaths) and
     ("data_batch_2" in allPaths) and
     ("data_batch_3" in allPaths) and
     ("data_batch_4" in allPaths) and
     ("data_batch_5" in allPaths) and
     ("test_batch" in allPaths)):
     
      question["score"] = question["max_score"]
      question["output"] = "CIFAR10 succesfully downloaded.  Points awarded."
  
  else:
    question["score"] = 0
    question["output"] = "I did not find the files for CIFAR10 in the output directory."


except Exception as e:
  question["score"] = 0
  question["output"] = "Your code resulted in an error, and so you were awarded no points.  Here is the error: " + str(e)
  
score = score + question["score"]
ret["tests"].append(question)


#================================
#================================
#QUESTION 2
#================================
#================================
print("\n=======================================================")
print("\nCommencing assessment of code submitted for question 2.")
question = {}
question["max_score"] = 10
question["name"] = "Test: Code succesfully creates a test/train split within a pickled object."
question["output"] = ""
question["score"] = 0

try:
  outPath = submission.dataSplit()
  print("I am attempting to load the pickle located at " + outPath + ": ")

  try:
    with open(outPath, "rb") as f:
      labData = pickle.load(f)
  except Exception as e:
    print("Something went wrong when I tried to load " + outPath + ".  Here is what I know: " + str(e))
    question["score"] = 0

  try:
    print("Pickle succesfully extracted.  Let's take a look inside.")
    print("Your dictionary has the keys: " + str(labData.keys()))

    if(("Y_train" in labData.keys()) or 
      ("Y_test" in labData.keys())):
      print(" ++++++++++ You capitalized the letter 'Y' in your variable name!  Heresy!  While this isn't totally unreasonable, the normal approach in almost all of the existing literature is to use a lower-case y, and thus I assume you used a lower-case y in my code (i.e., the code I use to generate your grade!).  Thus, to receive a passing grade, I humbly request you change your variables to 'y_test' and 'y_train', respectively. ++++++++++" )

    if(("X_test" in labData.keys()) and
      ("X_train" in labData.keys()) and
      ("y_test" in labData.keys()) and
      ("y_train" in labData.keys())):

      print("All dictionary keys are correct.  Checking the for shape validity.")
      print("Confirm we have the expected 50,000 training Images:")
      print(len(labData["X_train"]))
      print(len(labData["y_train"]))

      print("\nConfirm we have the expected 10,000 test Images:")
      print(len(labData["X_test"]))
      print(len(labData["y_test"]))

      print("\nConfirm the first image in our training set is a 32x32x3 matrix of values, representing a 32x32 image with 3 bands of color:")
      print(labData["X_train"][0].shape)

      print("\nConfirm our y observations make sense - i.e., an integer between 0 and 9 representing one of the 10 classes in CIFAR")
      print(max(labData["y_train"]))
      print(min(labData["y_train"]))

      if((len(labData["X_train"]) == 50000) and
         (len(labData["y_train"]) == 50000) and
         (len(labData["X_test"]) == 10000) and
         (len(labData["y_test"]) == 10000) and
         (str(labData["X_train"][0].shape) == "(32, 32, 3)") and
         (max(labData["y_train"]) == 9) and
         (min(labData["y_train"]) == 0)):

         question["score"] = question["max_score"]
         question["output"] = "All data checks passed!  Points awarded."
      
      else:
        question["score"] = 0
        question["output"] = "At least one of the checks for validity of your pickle failed!"
      
    else:
      question["score"] = 0
      question["output"] = "You seem to be missing one of X_train, y_train, X_test or y_test in the object you pickled."
             
  except Exception as e:
    question["score"] = 0
    question["output"] = "Your code resulted in an error, and so you were awarded no points.  Here is the error (ID-C): " + str(e)

except Exception as e:
  question["score"] = 0
  question["output"] = "Your code resulted in an error, and so you were awarded no points.  Here is the error (ID-D): " + str(e)

score = score + question["score"]
ret["tests"].append(question)

#================================
#================================
#QUESTION 3
#================================
#================================
print("\n=======================================================")
print("\nCommencing assessment of code submitted for question 3.")
question = {}
question["max_score"] = 5
question["name"] = "Implicit in this class is a definition of 'k' for the k-nearest-neighbors algorithm.  What is it?"
question["output"] = ""
question["score"] = 0

try:
  print("Your answer was: " + str(submission.questionThree()))
  if(submission.questionThree() == 1):
    question["score"] = question["max_score"]
    question["output"] = "Your answer, 1, was correct."

  else:
    question["score"] = 0
    question["output"] = "Your answer, " + str(submission.questionThree()) + " , is incorrect."

except:
  question["score"] = 0
  question["output"] = "You appear to be missing the function 'questionThree'."

score = score + question["score"]
ret["tests"].append(question)

#================================
#================================
#QUESTION 4
#================================
#================================
print("\n=======================================================")
print("\nCommencing assessment of code submitted for question 4.")
question = {}
question["max_score"] = 5
question["name"] = "What is the distance metric implemented?"
question["output"] = ""
question["score"] = 0

try:
  print("Your answer was: " + str(submission.questionFour()))
  if((submission.questionFour() == "L1")):
    question["score"] = question["max_score"]
    question["output"] = "Your answer, L1, was correct."

  else:
    question["score"] = 0
    question["output"] = "Your answer, " + str(submission.questionFour()) + " , is incorrect."
except:
  question["score"] = 0
  question["output"] = "You appear to be missing the function 'questionFour'."

score = score + question["score"]
ret["tests"].append(question)

#================================
#================================
#QUESTION 5
#================================
#================================
print("\n=======================================================")
print("\nCommencing assessment of code submitted for question 5.")
question = {}
question["max_score"] = 5
question["name"] = "If you run the below class with the first 100 test cases in the CIFAR dataset, approximately what accuracy do you receive?"
question["output"] = ""
question["score"] = 0

try:
  print("Your answer was: " + str(submission.questionFive()))
  if((submission.questionFive() == "30%")):
    question["score"] = question["max_score"]
    question["output"] = "Your answer, 30%, was correct."

  else:
    question["score"] = 0
    question["output"] = "Your answer, " + str(submission.questionFive()) + " , is incorrect."
except:
  question["score"] = 0
  question["output"] = "You appear to be missing the function 'questionFive'."
  
score = score + question["score"]
ret["tests"].append(question)


#================================
#================================
#QUESTION 6
#================================
#================================
print("\n=======================================================")
print("\nCommencing assessment of code submitted for question 6.")
question = {}
question["max_score"] = 20
question["name"] = "Implementing a KNN with a L2 Distance Metric."
question["output"] = ""
question["score"] = 0

X_train = labData["X_train"]
y_train = labData["y_train"]
X_test =  labData["X_test"]
y_test =  labData["y_test"]
y_test = y_test[:100]
X_test = X_test[:100]

#Classifier for comparison - saved as a Pickle
#So responses on Gradescope pop back quicker.
class comparisonClassifier:
     def __init__(self):
         pass

     def train(self, X, y):
         self.Xtr = X
         self.ytr = y
    
     def predict(self, X, k):
         Ypred = np.zeros(len(X), dtype=np.dtype(self.ytr.dtype))

         for i in range(0, len(X)):
             l2Distances = np.linalg.norm(self.Xtr - X[i])

             minimumIndices = np.argsort(l2Distances)
             kClosest = minimumIndices[:k]
             predClass, counts = np.unique(self.ytr[kClosest], return_counts=True)
             Ypred[i] = predClass[counts.argmax()]
        
         return Ypred

# knnClass = comparisonClassifier()
# knnClass.train(X = X_train, y = y_train)
# comparisonPredictKNN = knnClass.predict(X = X_test, k=10)

#with open("./comparisonKnnResult.pickle", 'wb') as f:
#    pickle.dump(comparisonPredictKNN, f)

print("I am loading my own KNN to compare to yours...")


comparisonPredictKNN = pickle.load(open("comparisonKnnResult.pickle", "rb"))

print("\nI am fitting the KNN you provided to see how it compares to my results.  Computing...")
try:
  studentKNNClassifier = submission.knnClassifier()
  studentKNNClassifier.train(X = X_train, y = y_train)
  studentPredictions = studentKNNClassifier.predict(X = X_test, k=10)

  #Difference between classifiers:
  agreementPercent = np.sum(comparisonPredictKNN == studentPredictions) / len(y_test)

  print("Our KNNs agreed " + str(agreementPercent*100) + " percent of the time. So, that is your grade for this question!")
  question["output"] = "Our KNNs agreed " + str(agreementPercent) + " percent of the time. So, that is your grade for this question!"
  question["score"] = agreementPercent * question["max_score"]

except Exception as e:
  print("\nI tried to fit your KNN, but it failed.  Here is what I know: " + str(e))
  question["score"] = 0

score = score + question["score"]
ret["tests"].append(question)


#================================
#================================
#QUESTION 7
#================================
#================================
print("\n=======================================================")
print("\nCommencing assessment of code submitted for question 7.")
question = {}
question["max_score"] = 20
question["name"] = "Implementing a Crossvalidation."
question["output"] = ""
question["score"] = 0

#====== Code for Comparison Algorithm =========
X_train = labData["X_train"]
y_train = labData["y_train"]
X_test =  labData["X_test"]
y_test =  labData["y_test"]
y_test = y_test[:1000]
X_test = X_test[:1000]
y_train = y_train[:1000]
X_train = y_train[:1000]

#Correct function:
#def crossFoldValidation(modelToValidate,
#                        X_train,
#                        y_train,
#                        hyperparameters={"k":5},
#                        folds=5):
    
#    k = hyperparameters["k"]

#    accuracies = []
#    X_folds = np.array_split(X_train, folds)
#    y_folds = np.array_split(y_train, folds)

#    for i in range(0,folds): 
#        classifier = modelToValidate
#        classifier.train(np.concatenate(np.delete(X_folds, [i], axis=0)), 
#                        np.concatenate(np.delete(y_folds, [i], axis=0)))
#        predictions = classifier.predict(X_folds[i], k=k)
#        correctCases = sum(predictions == y_folds[i])
#        accuracy = correctCases / len(y_folds[i])
#        accuracies.append(accuracy)

#    return(accuracies)  


#correctCrossfoldResults = crossFoldValidation(folds=5,
#                        modelToValidate = comparisonClassifier(),
#                        hyperparameters={"k":5},
#                        X_train = X_train,
#                        y_train = y_train)

#with open("./correctCrossfoldResults.pickle", 'wb') as f:
#    pickle.dump(correctCrossfoldResults, f)

print("I am loading my own cross validation to compare to yours...")


correctCrossfoldResults = pickle.load(open("correctCrossfoldResults.pickle", "rb"))
print("Here are the solutions I arrived at for each fold's accuracy: " + str(correctCrossfoldResults))

print("\nI am now fitting the cross validation you provided to see how it compares to my results.  Computing...")

try:
  studentCrossfoldResults = submission.crossFoldValidation(folds = 5,
                               modelToValidate = comparisonClassifier(),
                               X_train = X_train,
                               y_train = y_train)
  
  print("Here are the solutions I arrived at with your crossfold validation: " + str(studentCrossfoldResults))

  if(np.array_equal(correctCrossfoldResults, studentCrossfoldResults)):
    print("Your results and mine are identical!  Full credit awarded.")
    question["score"] = question["max_score"] 
    question["output"] = "Arrays matched, full credit awarded."


  else:
    print("Your answers are different from mine!  Looks like we're off by:")
    print(np.subtract(correctCrossfoldResults, studentCrossfoldResults))
    print("For a total error of:")
    print(np.sum(np.abs(np.subtract(correctCrossfoldResults, studentCrossfoldResults))))
    question["score"] = question["max_score"] / 2
    question["output"] = "Arrays did not match, but you're almost there.  Half credit awarded."

except Exception as e:
  print("\nI tried to fit your cross fold validation, but it failed.  Here is what I know: " + str(e))
  question["score"] = 0

score = score + question["score"]
ret["tests"].append(question)


#================================
#================================
#QUESTION 8
#================================
#================================
print("\n=======================================================")
print("\nCommencing assessment of code submitted for question 8.")
question = {}
question["max_score"] = 20
question["name"] = "Implementing Multiclass SVM Loss with Regularization."
question["output"] = ""
question["score"] = 0

y_train = labData["y_train"]
y_test =  labData["y_test"]
y_test = y_test[:250]
y_train = y_train[:250]

X_train = np.reshape(labData["X_train"][:250], (labData["X_train"][:250].shape[0], -1))
X_test = np.reshape(labData["X_test"][:250], (labData["X_test"][:250].shape[0], -1))

def svmClassifier(X, y, W, e, l):
    scores = X.dot(W)
    countTrainSamples = scores.shape[0]
    countClasses = scores.shape[1] 
    trueClassScores = scores[np.arange(scores.shape[0]), y]
    trueClassMatrix = np.matrix(trueClassScores).T 
    loss_ij = np.maximum(0, (scores - trueClassMatrix) + e) 
    loss_ij[np.arange(countTrainSamples), y] = 0
    correct = np.mean(np.equal(trueClassScores, np.amax(scores,axis=1)))
    dataLoss = np.sum(np.sum(loss_ij)) / countTrainSamples
    regLoss = np.sum(W*W)
    totalLoss = dataLoss + (l * regLoss)
    return({'dataLoss':dataLoss, 'regLoss':regLoss, 'totalLoss':totalLoss, 'percentCorrect':correct})


W = pickle.load(open("SVMInitWeights.pickle", "rb"))

correctResult = svmClassifier(X_train, y_train, W, e=1, l=1)

try:
  studentResults = submission.svmClassifier(X=X_train, 
                                            y=y_train, 
                                            W = W, 
                                            e = 1, 
                                            l = 1)
  try:
    if(studentResults["totalLoss"] == correctResult["totalLoss"]):
      print("Your outputs and mine match perfectly!  100%.")
      question["output"] = "SVM results matched perfectly, full credit awarded."
      question["score"] = question["max_score"]
    elif(studentResults["dataLoss"] == correctResult["dataLoss"]):
      print("Your data loss and mine agree, but not our reg or total loss.")
      print("Still, very close!  You get almost-full credit at this point.")
      question["score"] = question["max_score"]*0.9
    elif(studentResults["regLoss"] == correctResult["regLoss"]):
      print("Your regularization loss and mine agree, but not our data loss or total loss.")
      print("You're getting closer - half credit!")
      question["score"] = question["max_score"]*0.5

  except Exception as e:
    question["output"] = "I was able ot run your SVM, but it didn't return anything I could interpret.  Check my logs!"
    print("Something went wrong with my interpretation of your model output: " + str(e))
except Exception as e:
  print("I was unable to run your svmClassifier.  I am trying to invoke it with the following command: \n")
  print("  studentResults = submission.svmClassifier(X=X_train,")
  print("                                          y=y_train,")
  print("                                          W = W,")
  print("                                          e = 1,")
  print("                                          l = 1)")
  print("\n")
  print("With the following X and Y:")
  print('y_train = labData["y_train"]')
  print('y_test =  labData["y_test"]')
  print('y_test = y_test[:250]')
  print('y_train = y_train[:250]')
  print('X_train = np.reshape(labData["X_train"][:250], (labData["X_train"][:250].shape[0], -1))')
  print('X_test = np.reshape(labData["X_test"][:250], (labData["X_test"][:250].shape[0], -1))')
  print("\nI received the error: " + str(e))
  question["output"] = "Please see the log - I was unable to run your SVM!"


score = score + question["score"]
ret["tests"].append(question)

#================================
#================================
#QUESTION 9
#================================
#================================
print("\n=======================================================")
print("\nCommencing assessment of code submitted for question 9.")
question = {}
question["max_score"] = 20
question["name"] = "Implementing an optimizer for SVM."
question["output"] = ""
question["score"] = 0

y_train = labData["y_train"]
y_test =  labData["y_test"]
y_test = y_test[:1000]
y_train = y_train[:1000]

X_train = np.reshape(labData["X_train"][:1000], (labData["X_train"][:1000].shape[0], -1))
X_test = np.reshape(labData["X_test"][:1000], (labData["X_test"][:1000].shape[0], -1))

print("The weights I found resulted in a final image classification accuracy of about 30% (I'm going easy on you, and just running a few iterations).")
print("If you can get at least that good - or beat me - you get full credit!  Let's see how you did.")

print("Calculating....")
try:
  W = submission.svmOptimizer(X = X_train, y = y_train)
except Exception as e:
  print("I was unable to run your optimizer.  I tried to invoke it with: ")
  print(" W = submission.svmOptimizer(X = X_train, y = y_train)")
  print("Here is what I know: " + str(e))
try:
  res = svmClassifier(X_train, y_train, W, e=1, l=1)
  print("Your optimizer ran, and you got an accuracy of " + str(res['percentCorrect']*100) + "!")
  question['score'] = min(100.0, ((res['percentCorrect']*100) / 30)*100)
  print("Given that I got 30%, that means you get a score of " + str(question['score']))
except Exception as e:
  print("I was unable to use the weights from your optimizer.  I tried to invoke it with: ")
  print(" W = submission.svmOptimizer(X = X_train, y = y_train)")
  print(" res = svmClassifier(X_train, y_train, W, e=1, l=1")
  print("I have my own svmClassifier I want to test your weights with - my classifier is a correct solution")
  print("To the previous question.")
  print("Here are the weights I solved for with your function (submission.svmOptimizer): " + str(W))
  print("Here is what I know: " + str(e))
  question["output"] = "I ran into an error trying to run SVM with your optimizer weights.  Check out the logs."

score = score + question["score"]
ret["tests"].append(question)


#LEADERBOARD
ret["leaderboard"] = []

tim = {}
tim["name"] = "Runtime (seconds)"
tim["value"] = str(datetime.now() - startTime)
tim["order"] = "asc"
ret["leaderboard"].append(tim)

acc = {}
acc["name"] = "Accuracy (Percentage)"
acc["value"] = score / max_score
ret["leaderboard"].append(acc)

json.dumps(ret)
outF = open("/autograder/results/results.json", "w")
json.dump(ret, outF)
outF.close()
