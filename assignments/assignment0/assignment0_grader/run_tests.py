import json
import sys
from datetime import datetime

import submission

#Top level returns:
ret = {}
ret["output"] = "This was an example of how you will be graded in this course."
ret["visibility"] = "visible"
ret["stdout_visibility"] = "visible"
max_score = 50

#Code Tests:
ret["tests"] = []
score = 0

startTime = datetime.now()

#QUESTION 1

question = {}
question["max_score"] = 25
question["name"] = "Testing if your code succesfully adds two values together."
question["output"] = ""

try:
  val = submission.add(2,3)
  
  if(val == 5):
    question["score"] = 25
  else:
    question["score"] = 0
  
  question["output"] = "Your code provided the solution of 2+3 = " + str(val) + "."
except Exception as e:
  question["score"] = 0
  question["output"] = "Your code resulted in an error, and so you were awarded no points.  Here is the error: " + str(e)
  
score = score + question["score"]
ret["tests"].append(question)

#QUESTION 2 =====================
question = {}
question["max_score"] = 25
question["name"] = "Testing if your code succesfully multiplies two values together."
question["output"] = ""

try:
  val = submission.multiply(2,3)

  if(val == 6):
    question["score"] = 25
  else:
    question["score"] = 0
    
  question["output"] = "Your code provided the solution of 2*3 = " + str(val) + "."
except Exceptionn as e:
  question["score"] = 0
  question["output"] = "Your code resulted in an error, and so you were awarded no points.  Here is the error: " + str(e)

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
