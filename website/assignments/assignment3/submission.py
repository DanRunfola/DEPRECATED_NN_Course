##This toggles keras to only use CPUs.
##I highly recommend you turn this off -
#for example only!
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import keras

#For assignment 3, you will be creating a few different models
#and uploading your solutions in the form of *.h5 files.
#Each model is limited to be no more than 100 megabytes in total.
#Unlike previous assignments, you will not be provided with
#either data or pre-written code: it's up to you to find a labeled dataset
#and implement a model!

#Your model will then be tested against my own labeled data, which I have sourced
#from a wide range of websites + personal databases.  You will not have access 
#to those databases.

#You will upload a h5 file for each question - i.e.,
#"Q1.h5" would be your upload for Q1.

#For every question, the code I run to test your model performance 
#will be stated.

#Note for each submission, you must include the accuracy metric prescribed or you will not 
#be awarded any points. The rules for each submission are noted below.

#Note that functions must ALWAYS be named exactly as noted, or you will not be 
#awarded any points. 

#Because the number of points you are awarded is directly proportional to the score you get
#(i.e., if you get 80% accuracy on a question, you get 80% of the points for that question),
#a curve will be applied to this assignment so that the student with the best models
#receives a 100, and all other scores are increased accordingly.

#So you know where you stand, the leaderboard on gradescope will show the scores for each student.
#It is recommended you use an anonymous handle on gradescope (you enter it when you submit)!

#The curve will be implemented once based on total score, not on a quesiton-by-question basis.

#All models must be built and submitted using Keras, as per the examples.  Any preprocessing
#other than what I apply in the data loader must be done within the network.

#========================================
#========================================
#EXAMPLE NET
#========================================
#========================================
#I re-use this example net throughout this submission file.
#You should *definitely* be creating your own net for each
#problem!

def exampleNet(inputShape, outputClasses, accMetrics):
    m = keras.models.Sequential()
    m.add(keras.layers.Conv2D(filters=64,
                              kernel_size=(4,4),
                              activation="tanh",
                              input_shape=inputShape))
    m.add(keras.layers.GlobalAveragePooling2D())
    m.add(keras.layers.Dense(units=outputClasses))
    m.compile(optimizer=keras.optimizers.SGD(learning_rate=.001),
                                            loss='categorical_hinge',
                                            metrics=accMetrics)
    
    return(m)

#=========================================
#=========================================
#LAB QUESTION 1
#=========================================
#=========================================
#FILE NAME: "Q1.h5"
#CHALLENGE: Write an algorithm that will estimate a binary outcome: 
#           if a street sign is a stop sign, or not.  Input images being tested
#           will be images of street signs of variable dimensions, and the sign
#           will always be a prominent eleemnt of the image (though the image may not always
#           be perfectly cropped).
#           Images being tested will be American stop signs (red, octagon).

#You *must* include metrics=['categorical_accuracy'] in your
#compilation.

#Validation Code:
#studentModel = keras.models.load_model("/autograder/submission/Q1.h5")
#dataGenerator = keras.preprocessing.image.ImageDataGenerator(samplewise_center=True)
#test = dataGenerator.flow_from_directory(imagePath, class_mode='categorical', batch_size=32, target_size=(64, 64))
#modelOutcome = studentModel.evaluate(test)

dataGenerator = keras.preprocessing.image.ImageDataGenerator(samplewise_center=True)
train = dataGenerator.flow_from_directory("./submissionExamples/streetSigns", class_mode='categorical', batch_size=2, target_size=(64, 64))
    
model = exampleNet(inputShape=(64,64,3), outputClasses=2, accMetrics=['categorical_accuracy'])
model.fit(train)
model.save("./submissionExamples/models/Q1.h5")

#=========================================
#=========================================
#LAB QUESTION 2
#=========================================
#=========================================
#FILE NAME: "Q2.h5"
#CHALLENGE: Write an algorithm that will correctly classify images of bugs
#           into one of three classes: butterfly, ant, and caterpillar.
#           Input images being tested will be of both posed and wild cases,
#           the bug will always be a prominent feature of the image.

#You *must* include metrics=['categorical_accuracy'] in your
#compilation.

#Validation Code:
#studentModel = keras.models.load_model("/autograder/submission/Q2.h5")
#dataGenerator = keras.preprocessing.image.ImageDataGenerator(samplewise_center=True)
#test = dataGenerator.flow_from_directory(imagePath, class_mode='categorical', batch_size=32, target_size=(64, 64))
#modelOutcome = studentModel.evaluate(test)

dataGenerator = keras.preprocessing.image.ImageDataGenerator(samplewise_center=True)
train = dataGenerator.flow_from_directory("./submissionExamples/bugs", class_mode='categorical', batch_size=3, target_size=(64, 64))
    
model = exampleNet(inputShape=(64,64,3), outputClasses=3, accMetrics=['categorical_accuracy'])
model.fit(train)
model.save("./submissionExamples/models/Q2.h5")


#=========================================
#=========================================
#LAB QUESTION 3
#=========================================
#=========================================
#FILE NAME: "Q3.h5"
#CHALLENGE: Write an algorithm that will correctly identify
#           the occurence of a natural disaster from aerial imagery.
#           Your output should be a classificaiton which distinguishes
#           between three cases: normal (no disaster), active fire/smoke, and building collapse.

#You *must* include metrics=['categorical_accuracy'] in your
#compilation.

#Validation Code:
#studentModel = keras.models.load_model("/autograder/submission/Q3.h5")
#dataGenerator = keras.preprocessing.image.ImageDataGenerator(samplewise_center=True)
#test = dataGenerator.flow_from_directory(imagePath, class_mode='categorical', batch_size=16, target_size=(256, 256))
#modelOutcome = studentModel.evaluate(test)

dataGenerator = keras.preprocessing.image.ImageDataGenerator(samplewise_center=True)
train = dataGenerator.flow_from_directory("./submissionExamples/disasters", class_mode='categorical', batch_size=3, target_size=(256, 256))
    
model = exampleNet(inputShape=(64,64,3), outputClasses=3, accMetrics=['categorical_accuracy'])
model.fit(train)
model.save("./submissionExamples/models/Q3.h5")