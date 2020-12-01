#In this example assignment, you must code a calculator with
#functions which multiply and add.  

def multiply(x, y):
  return(x * y)

#Intentionally wrong addition function!
def add(x, y):
  return(x + y + str("error"))

  
  
if __name__ == '__main__':
  print(multiply(2,3))
  print(add(1,2))