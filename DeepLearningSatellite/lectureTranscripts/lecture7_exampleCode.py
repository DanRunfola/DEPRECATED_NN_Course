# pylint: skip-file

import numpy as np
import matplotlib.pyplot as plt

class simpleNeuralNetwork():
    def forwardPass(W,X):
        for node in computationalGraph:
            node.calculation()
        return totalLoss
    
    def backwardPass():
        for node in computationalGraph.flip():
            node.gradients()
        
        return W_and_X_gradients

        
################################

class MultiplicationNode():
    def forwardPass(W,X):
        output = X * W
        return output
    

class MultiplicationNode():
    def forwardPass(input1,input2):
        output = input1 * input2
        self.input1 = input1
        self.input2 = input2
        return output

    def backwardPass(dOutput):
        dInput1 = self.input2 * dOutput
        dInput2 = self.input1 * dOutput
        return [dInput1, dInput2]

        