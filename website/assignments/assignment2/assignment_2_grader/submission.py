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