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

#You will upload a h5 file just like in the last assignment for each question - i.e.,
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

#All models must be built and submitted using Keras, as per the examples.

#========================================
#========================================
#EXAMPLE NET
#========================================
#========================================
#I re-use this example net throughout this submission file.
#You should *definitely* be creating your own net for each
#problem!

def exampleNet(inputShape, outputClasses):
    m = keras.models.Sequential()
    m.add(keras.layers.Conv2D(filters=64,
                              kernel_size=(4,4),
                              activation="tanh",
                              input_shape=inputShape))
    m.add(keras.layers.GlobalAveragePooling2D())
    m.add(keras.layers.Dense(units=outputClasses))
    m.compile(optimizer=keras.optimizers.SGD(learning_rate=.001),
                                            loss='categorical_hinge',
                                            metrics=['categorical_accuracy'])
    
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
#compilation (i.e., see below).

#Validation Code:
#studentModel = keras.models.load_model("/autograder/submission/Q1.h5")
#dataGenerator = keras.preprocessing.image.ImageDataGenerator(samplewise_center=True)
#test = dataGenerator.flow_from_directory(imagePath, class_mode='categorical', batch_size=32, target_size=(64, 64))
#modelOutcome = studentModel.evaluate(test)

dataGenerator = keras.preprocessing.image.ImageDataGenerator(samplewise_center=True)
train = dataGenerator.flow_from_directory("./submissionExamples/streetSigns", class_mode='categorical', batch_size=2, target_size=(64, 64))
    
model = exampleNet(inputShape=(64,64,3), outputClasses=2)
model.fit(train)
model.save("./submissionExamples/models/Q1.h5")