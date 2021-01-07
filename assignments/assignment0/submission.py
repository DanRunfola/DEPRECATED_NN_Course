#In this example assignment, you must code a calculator with
#functions which multiply and add.  
#You will submit your version of this file to gradescope.com.
#It will then automatically be tested and graded.
#Note that your function names MUST be what is specified
#in the submission file - i.e., the multiply function
#must be named "multiply".
#Additionally, you must name your file submission.py .
#Note you can upload this file to test - gradescope
#will allow you to upload as many trial solutions
#as you want, and grade them automatically!
#Uploaded as-is you will get a 0, but you can then 
#upload again your own modified code.
#Generally, on real labs, I recommend you work
#one question at a time - otherwise you'll be debugging code
#forever.
#Also, you should be making sure your code runs on your
#local computer before uploading everything,
#as the autograder can take minutes to run in some cases
#(up to 10 minutes).


#Example function, worth no points:
def subtract(x, y):
  return(x - y)

#=========================================
#=========================================
#LAB QUESTION 1
#=========================================
#=========================================
#FUNCTION NAME: multiply 
#FUNCTION DESCRIPTION: Function to multiply two numbers
#FUNCTION OUTPUT: Float of the two numbers multiplied
#FUNCTION NOTES: The default values below are incorrect.  You should
#change the return line to be correct.

def multiply(x, y):
  return(x - y * x ^ y)


#=========================================
#=========================================
#LAB QUESTION 2
#=========================================
#=========================================
#FUNCTION NAME: add 
#FUNCTION DESCRIPTION: Function to add two numbers
#FUNCTION OUTPUT: Float of the two numbers added
#FUNCTION NOTES: The default values below are incorrect.  You should
#change the return line to be correct.
def add(x, y):
  return(x + y + str("error"))


#The below is optional, but helpful for you to test your code:  

if __name__ == '__main__':
  print(multiply(2,3))
  print(add(1,2))